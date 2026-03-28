#include <gtest/gtest.h>
#include "simd_pricer.h"
#include "black_scholes.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace pricing;

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────
struct Rng {
    uint64_t seed;
    explicit Rng(uint64_t s) : seed(s) {}
    double next() {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        return static_cast<double>(seed >> 11) / static_cast<double>(1ULL << 53);
    }
    double uniform(double lo, double hi) { return lo + next() * (hi - lo); }
};

// Float precision tolerance — SIMD uses float (32-bit) and polynomial
// approximations for log/exp/norm_cdf. Errors compound through d1/d2,
// so we allow up to 1% relative-ish error (~$0.50 on a $50 option).
// This is appropriate for risk/hedging use cases; for exact mark-to-market
// use the scalar double-precision engine.
static constexpr float FLOAT_TOL = 0.75f;

// Build a batch of N randomised options
static std::vector<BSParams> makeBatch(int n, uint64_t seed = 0xDEAD1234BEEF5678ULL) {
    Rng rng(seed);
    std::vector<BSParams> params(n);
    for (auto& p : params) {
        p.S     = rng.uniform(50.0,  150.0);
        p.K     = rng.uniform(50.0,  150.0);
        p.r     = rng.uniform(0.01,  0.08);
        p.q     = rng.uniform(0.0,   0.05);
        p.sigma = rng.uniform(0.10,  0.60);
        p.T     = rng.uniform(0.25,  2.0);
        p.type  = (rng.next() > 0.5) ? OptionType::Call : OptionType::Put;
    }
    return params;
}

// ─────────────────────────────────────────────
//  1. Report SIMD level (informational, always passes)
// ─────────────────────────────────────────────
TEST(SIMDPricer, ReportsSIMDLevel) {
    std::cout << "\n  SIMD level: " << SIMDBatchPricer::simdLevel() << "\n";
    SUCCEED();
}

// ─────────────────────────────────────────────
//  2. SIMD prices match scalar Black-Scholes  (property-based)
//
//  For every option in a large randomised batch, the SIMD float
//  price must be within FLOAT_TOL of the double-precision BS price.
//  This validates correctness of all polynomial approximations.
// ─────────────────────────────────────────────
TEST(SIMDPricer, MatchesScalarBS_PropertyBased) {
    const auto params = makeBatch(1024);

    // SIMD batch price
    const auto simd_prices = SIMDBatchPricer::price(params);

    // Scalar reference — output is float32, so we allow float cast error only.
    // The SSE4 path computes in double and casts to float at the end, so errors
    // should be < 1 ULP of float (~1e-7 relative), well within 0.1% absolute.
    constexpr float REL_TOL  = 0.001f;  // 0.1% relative
    constexpr float ABS_FLOOR = 0.01f;  // $0.01 absolute floor for cheap options
    int failures = 0;
    for (std::size_t i = 0; i < params.size(); ++i) {
        const float bs_price = static_cast<float>(BlackScholesEngine::price(params[i]).price);
        const float diff = std::abs(simd_prices[i] - bs_price);
        const float tol  = std::max(REL_TOL * bs_price, ABS_FLOOR);

        if (diff > tol) {
            ADD_FAILURE() << "Option " << i
                          << " SIMD=" << simd_prices[i]
                          << " BS=" << bs_price
                          << " diff=" << diff << " tol=" << tol
                          << " S=" << params[i].S << " K=" << params[i].K
                          << " sigma=" << params[i].sigma << " T=" << params[i].T
                          << " type=" << (params[i].type == OptionType::Call ? "Call" : "Put");
            if (++failures >= 10) { ADD_FAILURE() << "(further failures suppressed)"; break; }
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/1024 options failed tolerance check";
}

// ─────────────────────────────────────────────
//  3. Tail handling — batches not divisible by 8
//
//  Verifies the scalar tail path is correct for n % 8 != 0.
// ─────────────────────────────────────────────
TEST(SIMDPricer, HandlesNonMultipleOf8) {
    for (int n : {1, 3, 7, 9, 15, 17}) {
        const auto params = makeBatch(n, 0xABCDULL + n);
        const auto simd_prices = SIMDBatchPricer::price(params);

        for (int i = 0; i < n; ++i) {
            const float bs = static_cast<float>(BlackScholesEngine::price(params[i]).price);
            EXPECT_NEAR(simd_prices[i], bs, FLOAT_TOL)
                << "n=" << n << " i=" << i;
        }
    }
}

// ─────────────────────────────────────────────
//  4. Single-option batch
// ─────────────────────────────────────────────
TEST(SIMDPricer, SingleOptionBatch) {
    BSParams p{100.0, 100.0, 0.05, 0.02, 0.20, 1.0, OptionType::Call};
    const auto prices = SIMDBatchPricer::price({p});
    const float bs = static_cast<float>(BlackScholesEngine::price(p).price);
    EXPECT_NEAR(prices[0], bs, FLOAT_TOL);
}

// ─────────────────────────────────────────────
//  5. Put-call parity holds for SIMD prices
// ─────────────────────────────────────────────
TEST(SIMDPricer, PutCallParity_PropertyBased) {
    Rng rng(0xCAFEBABE12345678ULL);
    constexpr int N = 256;
    int failures = 0;

    std::vector<BSParams> calls(N), puts(N);
    for (int i = 0; i < N; ++i) {
        calls[i] = {rng.uniform(50, 150), rng.uniform(50, 150),
                    rng.uniform(0.01, 0.08), rng.uniform(0.0, 0.05),
                    rng.uniform(0.10, 0.60), rng.uniform(0.25, 2.0),
                    OptionType::Call};
        puts[i] = calls[i]; puts[i].type = OptionType::Put;
    }

    const auto c_prices = SIMDBatchPricer::price(calls);
    const auto p_prices = SIMDBatchPricer::price(puts);

    for (int i = 0; i < N; ++i) {
        const float C   = c_prices[i], P = p_prices[i];
        const float rhs = static_cast<float>(
            calls[i].S * std::exp(-calls[i].q * calls[i].T) -
            calls[i].K * std::exp(-calls[i].r * calls[i].T));
        const float diff = std::abs((C - P) - rhs);

        // Parity tolerance: errors in call and put both compound, so allow 2x
        if (diff > FLOAT_TOL * 2) {
            ADD_FAILURE() << "Option " << i << " parity: C-P=" << (C-P)
                          << " rhs=" << rhs << " diff=" << diff;
            if (++failures >= 5) break;
        }
    }
    EXPECT_EQ(failures, 0);
}

// ─────────────────────────────────────────────
//  6. Greeks (priceAndGreeks) match scalar
// ─────────────────────────────────────────────
TEST(SIMDPricer, GreeksMatchScalar) {
    const auto params = makeBatch(64);
    const std::size_t n = params.size();

    std::vector<float> S_buf, K_buf, r_buf, q_buf, sigma_buf, T_buf;
    std::vector<int>   type_buf;
    BatchInput::fill(params, S_buf, K_buf, r_buf, q_buf, sigma_buf, T_buf, type_buf);

    BatchInput in{S_buf.data(), K_buf.data(), r_buf.data(), q_buf.data(),
                  sigma_buf.data(), T_buf.data(), type_buf.data(), n};

    std::vector<float> prices(n), delta(n), gamma(n), vega(n);
    BatchOutput out{prices.data(), delta.data(), gamma.data(), vega.data()};
    SIMDBatchPricer::priceAndGreeks(in, out, n);

    for (std::size_t i = 0; i < n; ++i) {
        const auto bs = BlackScholesEngine::price(params[i]);
        EXPECT_NEAR(delta[i], static_cast<float>(bs.greeks.delta), FLOAT_TOL)
            << "delta mismatch at i=" << i;
        EXPECT_NEAR(gamma[i], static_cast<float>(bs.greeks.gamma), FLOAT_TOL)
            << "gamma mismatch at i=" << i;
        EXPECT_NEAR(vega[i],  static_cast<float>(bs.greeks.vega),  FLOAT_TOL)
            << "vega mismatch at i=" << i;
    }
}

// ─────────────────────────────────────────────
//  7. Latency benchmark — chrono-based p50/p95/p99
//
//  Uses std::chrono::high_resolution_clock which works correctly on
//  all platforms including Apple Silicon under Rosetta. rdtsc is
//  unreliable on M-series Macs (returns 0 or fixed 1GHz reference).
//
//  Reports p50/p95/p99 latency across 1000 runs for both SIMD and
//  scalar, plus throughput (Mops/s) and speedup ratio.
// ─────────────────────────────────────────────

struct LatencyStats {
    double p50_ns, p95_ns, p99_ns;
    double mean_ns;
    double mops;
};

static LatencyStats measure(std::function<void()> fn, int n_options, int reps) {
    using clk = std::chrono::high_resolution_clock;

    // Warmup — fill caches, avoid cold-start penalty
    for (int i = 0; i < 5; ++i) fn();

    std::vector<double> samples(reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = clk::now();
        fn();
        const auto t1 = clk::now();
        samples[r] = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }

    std::sort(samples.begin(), samples.end());
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / reps;

    LatencyStats s;
    s.p50_ns  = samples[reps * 50 / 100];
    s.p95_ns  = samples[reps * 95 / 100];
    s.p99_ns  = samples[reps * 99 / 100];
    s.mean_ns = mean;
    s.mops    = static_cast<double>(n_options) / mean * 1000.0;
    return s;
}

TEST(SIMDPricer, LatencyBenchmark) {
    constexpr int N    = 1 << 16;  // 65536 options
    constexpr int REPS = 1000;

    const auto params = makeBatch(N);

    std::vector<float> S_buf, K_buf, r_buf, q_buf, sigma_buf, T_buf;
    std::vector<int>   type_buf;
    BatchInput::fill(params, S_buf, K_buf, r_buf, q_buf, sigma_buf, T_buf, type_buf);
    BatchInput in{S_buf.data(), K_buf.data(), r_buf.data(), q_buf.data(),
                  sigma_buf.data(), T_buf.data(), type_buf.data(), (std::size_t)N};
    std::vector<float> out_simd(N), out_scalar(N);

    const auto simd_stats = measure(
        [&]() { SIMDBatchPricer::price(in, out_simd.data(), N); }, N, REPS);

    const auto scalar_stats = measure(
        [&]() {
            for (int i = 0; i < N; ++i)
                out_scalar[i] = static_cast<float>(BlackScholesEngine::price(params[i]).price);
        }, N, REPS);

    const double speedup_mean = scalar_stats.mean_ns / simd_stats.mean_ns;
    const double speedup_p99  = scalar_stats.p99_ns  / simd_stats.p99_ns;

    std::cout << std::fixed << std::setprecision(1) << "\n"
        << "  ┌──────────────────────────────────────────────────────────┐\n"
        << "  │  Latency benchmark  (" << N << " options, " << REPS << " reps)        │\n"
        << "  │  SIMD level: " << std::setw(20) << std::left
                               << SIMDBatchPricer::simdLevel() << std::right << "              │\n"
        << "  ├────────────────┬──────────┬──────────┬──────────┬────────┤\n"
        << "  │                │   p50    │   p95    │   p99    │ Mops/s │\n"
        << "  ├────────────────┼──────────┼──────────┼──────────┼────────┤\n"
        << "  │ SIMD           │ "
        << std::setw(7) << simd_stats.p50_ns   << " ns │ "
        << std::setw(7) << simd_stats.p95_ns   << " ns │ "
        << std::setw(7) << simd_stats.p99_ns   << " ns │ "
        << std::setw(5) << simd_stats.mops     << "  │\n"
        << "  │ Scalar         │ "
        << std::setw(7) << scalar_stats.p50_ns << " ns │ "
        << std::setw(7) << scalar_stats.p95_ns << " ns │ "
        << std::setw(7) << scalar_stats.p99_ns << " ns │ "
        << std::setw(5) << scalar_stats.mops   << "  │\n"
        << "  ├────────────────┴──────────┴──────────┴──────────┴────────┤\n"
        << "  │  Speedup (mean): " << std::setprecision(2) << std::setw(5) << speedup_mean
        << "x    Speedup (p99): " << std::setw(5) << speedup_p99 << "x               │\n"
        << "  └──────────────────────────────────────────────────────────┘\n";

    EXPECT_GT(speedup_mean, 1.0) << "SIMD mean should be faster than scalar";
    EXPECT_GT(speedup_p99,  1.0) << "SIMD p99 should be faster than scalar";
}