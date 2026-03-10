#include <gtest/gtest.h>
#include "black_scholes.h"

#include <cmath>

using namespace pricing;

// ─────────────────────────────────────────────
//  Test fixtures & helpers
// ─────────────────────────────────────────────
static constexpr double GREEK_TOL  = 1e-5;   // greek precision
static constexpr double IV_TOL     = 1e-6;   // IV round-trip tolerance

// Canonical ATM params used across many tests
static BSParams atm() {
    return {100.0, 100.0, 0.05, 0.02, 0.20, 1.0, OptionType::Call};
}

// Separate bump sizes per Greek — first- and second-order FD have different
// optimal step sizes: h1 ~ eps^(1/2)*scale, h2 ~ eps^(1/3)*scale.
static constexpr double BUMP_S_DELTA = 0.001;  // $0.10 — first-order (delta)
static constexpr double BUMP_S_GAMMA = 0.1;    // $0.10 — second-order (gamma)
static constexpr double BUMP_SIGMA   = 0.01;   // 1 vol point — first-order (vega)
static constexpr double BUMP_R       = 0.0001; // 1bp rate — first-order (rho)

// ─────────────────────────────────────────────
//  1. Put-Call Parity (property-based)
//
//  C - P = S*e^{-qT} - K*e^{-rT}
//
//  Verified across 500 randomised parameter combinations covering
//  a wide range of moneyness, vol, rate, and maturity regimes.
//  Tolerance is 1e-10 -- this is an exact algebraic identity in
//  Black-Scholes, not an approximation, so it holds to near
//  machine precision regardless of inputs.
// ─────────────────────────────────────────────
TEST(BlackScholes, PutCallParityPropertyBased) {
    // Xorshift64 -- deterministic, reproducible, no <random> dependency
    uint64_t seed = 0xDEADBEEFCAFEBABEULL;
    auto next = [&]() -> double {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return static_cast<double>(seed >> 11) / static_cast<double>(1ULL << 53);
    };
    auto uniform = [&](double lo, double hi) { return lo + next() * (hi - lo); };

    constexpr int    N_TRIALS = 500;
    constexpr double TOL      = 1e-10;

    int failures = 0;
    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = uniform(10.0,   500.0);
        const double K     = uniform(10.0,   500.0);
        const double r     = uniform(0.0,    0.15);
        const double q     = uniform(0.0,    0.10);
        const double sigma = uniform(0.05,   1.50);
        const double T     = uniform(1.0/365.0, 5.0);

        BSParams call_p{S, K, r, q, sigma, T, OptionType::Call};
        BSParams put_p {S, K, r, q, sigma, T, OptionType::Put };

        const double C   = BlackScholesEngine::price(call_p).price;
        const double P   = BlackScholesEngine::price(put_p ).price;
        const double lhs = C - P;
        const double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);

        if (std::abs(lhs - rhs) > TOL) {
            ADD_FAILURE() << "Put-call parity violated on trial " << i << ":"
                          << "  S=" << S << " K=" << K << " r=" << r
                          << " q=" << q << " sigma=" << sigma << " T=" << T << ""
                          << "  C-P=" << lhs << "  rhs=" << rhs
                          << "  diff=" << std::abs(lhs - rhs);
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed";
}

// ─────────────────────────────────────────────
//  2. Known reference prices  (Haug tables)
// ─────────────────────────────────────────────
TEST(BlackScholes, KnownCallPrice) {
    // Reference: Haug "Option Pricing Formulas" 2nd ed., Table 1-1
    // S=60, K=65, r=8%, q=0, sigma=30%, T=0.25  => call ~ 2.1334
    BSParams p{60.0, 65.0, 0.08, 0.0, 0.30, 0.25, OptionType::Call};
    EXPECT_NEAR(BlackScholesEngine::price(p).price, 2.1334, 1e-4);
}

TEST(BlackScholes, KnownPutPrice) {
    // Same params via put-call parity
    BSParams p{60.0, 65.0, 0.08, 0.0, 0.30, 0.25, OptionType::Put};
    const double expected = 2.1334
        + 65.0 * std::exp(-0.08 * 0.25)
        - 60.0;
    EXPECT_NEAR(BlackScholesEngine::price(p).price, expected, 1e-4);
}

// ─────────────────────────────────────────────
//  3. Boundary / limit behaviour
// ─────────────────────────────────────────────
TEST(BlackScholes, DeepITMCallApproachesForwardPrice) {
    // Very low strike => call price ~ S*e^{-qT} - K*e^{-rT}
    BSParams p{100.0, 1.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call};
    const double forward = 100.0 - 1.0 * std::exp(-0.05 * 1.0);
    EXPECT_NEAR(BlackScholesEngine::price(p).price, forward, 1e-2);
}

TEST(BlackScholes, DeepOTMCallApproachesZero) {
    BSParams p{100.0, 100000.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call};
    EXPECT_NEAR(BlackScholesEngine::price(p).price, 0.0, 1e-10);
}

TEST(BlackScholes, ZeroVolCallEqualsIntrinsic) {
    // sigma -> 0 => price = max(S*e^{-qT} - K*e^{-rT}, 0)
    BSParams p{110.0, 100.0, 0.05, 0.0, 1e-9, 1.0, OptionType::Call};
    const double intrinsic = 110.0 - 100.0 * std::exp(-0.05 * 1.0);
    EXPECT_NEAR(BlackScholesEngine::price(p).price, intrinsic, 1e-4);
}

// ─────────────────────────────────────────────
//  4. Greeks — sign & magnitude sanity checks
// ─────────────────────────────────────────────
TEST(BlackScholes, CallDeltaBetweenZeroAndOne) {
    auto p = atm();
    const double d = BlackScholesEngine::delta(p);
    EXPECT_GT(d, 0.0);
    EXPECT_LT(d, 1.0);
}

TEST(BlackScholes, PutDeltaBetweenNegOneAndZero) {
    auto p = atm(); p.type = OptionType::Put;
    const double d = BlackScholesEngine::delta(p);
    EXPECT_LT(d, 0.0);
    EXPECT_GT(d, -1.0);
}

TEST(BlackScholes, GammaIsPositive) {
    EXPECT_GT(BlackScholesEngine::gamma(atm()), 0.0);
}

TEST(BlackScholes, ThetaIsNegativeForLongOption) {
    EXPECT_LT(BlackScholesEngine::theta(atm()), 0.0);
}

TEST(BlackScholes, VegaIsPositive) {
    EXPECT_GT(BlackScholesEngine::vega(atm()), 0.0);
}

TEST(BlackScholes, CallRhoIsPositive) {
    EXPECT_GT(BlackScholesEngine::rho(atm()), 0.0);
}

TEST(BlackScholes, PutRhoIsNegative) {
    auto p = atm(); p.type = OptionType::Put;
    EXPECT_LT(BlackScholesEngine::rho(p), 0.0);
}

// ─────────────────────────────────────────────
//  5. Greeks vs. finite difference
// ─────────────────────────────────────────────
TEST(BlackScholes, DeltaMatchesFiniteDifference) {
    auto p = atm();
    auto p_up = p; p_up.S += BUMP_S_DELTA;
    auto p_dn = p; p_dn.S -= BUMP_S_DELTA;

    const double fd_delta = (BlackScholesEngine::price(p_up).price
                           - BlackScholesEngine::price(p_dn).price)
                           / (2.0 * BUMP_S_DELTA);
    EXPECT_NEAR(BlackScholesEngine::delta(p), fd_delta, GREEK_TOL);
}

TEST(BlackScholes, GammaMatchesFiniteDifference) {
    auto p = atm();
    auto p_up = p; p_up.S += BUMP_S_GAMMA;
    auto p_dn = p; p_dn.S -= BUMP_S_GAMMA;

    const double fd_gamma = (BlackScholesEngine::price(p_up).price
                           - 2.0 * BlackScholesEngine::price(p).price
                           + BlackScholesEngine::price(p_dn).price)
                           / (BUMP_S_GAMMA * BUMP_S_GAMMA);
    EXPECT_NEAR(BlackScholesEngine::gamma(p), fd_gamma, GREEK_TOL);
}

TEST(BlackScholes, VegaMatchesFiniteDifference) {
    auto p = atm();
    auto p_up = p; p_up.sigma += BUMP_SIGMA;
    auto p_dn = p; p_dn.sigma -= BUMP_SIGMA;

    // Analytic vega is per 1% vol move (per 0.01 sigma).
    // FD: divide by 2h to get dV/d(sigma), then multiply by 0.01 to match the per-1% scaling.
    const double fd_vega = (BlackScholesEngine::price(p_up).price
                          - BlackScholesEngine::price(p_dn).price)
                          / (2.0 * BUMP_SIGMA) * 0.01;
    EXPECT_NEAR(BlackScholesEngine::vega(p), fd_vega, 1e-4);
}

TEST(BlackScholes, RhoMatchesFiniteDifference) {
    auto p = atm();
    auto p_up = p; p_up.r += BUMP_R;
    auto p_dn = p; p_dn.r -= BUMP_R;

    const double fd_rho = (BlackScholesEngine::price(p_up).price
                         - BlackScholesEngine::price(p_dn).price)
                         / (2.0 * BUMP_R) / 100.0;
    EXPECT_NEAR(BlackScholesEngine::rho(p), fd_rho, GREEK_TOL);
}

// ─────────────────────────────────────────────
//  6. Implied Volatility — round-trip (property-based)
//
//  For any valid (S, K, r, q, sigma, T), pricing an option then
//  solving for IV must recover the original sigma exactly.
//  Tested across 500 randomised trials covering both calls and puts,
//  spanning deep ITM/OTM, short/long dated, and low/high vol regimes.
//
//  Skipped trials: deep OTM options with very short maturity produce
//  prices so close to zero that the IV solver has no meaningful signal
//  to invert — these are noted and skipped rather than failed.
// ─────────────────────────────────────────────
TEST(BlackScholes, IVRoundTripPropertyBased) {
    uint64_t seed = 0xC0FFEE12345678ABULL;
    auto next = [&]() -> double {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return static_cast<double>(seed >> 11) / static_cast<double>(1ULL << 53);
    };
    auto uniform = [&](double lo, double hi) { return lo + next() * (hi - lo); };

    constexpr int    N_TRIALS  = 500;
    constexpr double IV_TOL_PT = 1e-6;   // round-trip tolerance: recover sigma to 1e-6
    constexpr double MIN_PRICE = 1e-6;   // skip trials where option price is effectively zero

    int failures = 0;
    int skipped  = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = uniform(20.0,   300.0);
        const double K     = uniform(20.0,   300.0);
        const double r     = uniform(0.0,    0.10);
        const double q     = uniform(0.0,    0.08);
        const double sigma = uniform(0.05,   1.00);   // 5% to 100% vol
        const double T     = uniform(1.0/52.0, 3.0);  // 1 week to 3 years
        // Alternate call/put on each trial to cover both types
        const OptionType type = (i % 2 == 0) ? OptionType::Call : OptionType::Put;

        BSParams p{S, K, r, q, sigma, T, type};
        const double mkt_price = BlackScholesEngine::price(p).price;

        // Skip near-zero prices — IV is numerically undefined here
        if (mkt_price < MIN_PRICE) {
            ++skipped;
            continue;
        }

        try {
            const double iv = BlackScholesEngine::impliedVol(
                mkt_price, S, K, r, q, T, type);

            if (std::abs(iv - sigma) > IV_TOL_PT) {
                ADD_FAILURE() << "IV round-trip failed on trial " << i << ":\n"
                              << "  type="  << (type == OptionType::Call ? "Call" : "Put")
                              << " S=" << S << " K=" << K << " r=" << r
                              << " q=" << q << " sigma=" << sigma << " T=" << T << "\n"
                              << "  mkt_price=" << mkt_price
                              << "  solved_iv=" << iv
                              << "  diff=" << std::abs(iv - sigma);
                ++failures;
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << "IV solver threw on trial " << i << ": " << e.what() << "\n"
                          << "  S=" << S << " K=" << K << " sigma=" << sigma << " T=" << T;
            ++failures;
        }
    }

    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed ("
                           << skipped << " skipped — price below " << MIN_PRICE << ")";
}

// ─────────────────────────────────────────────
//  7. Input validation
// ─────────────────────────────────────────────
TEST(BlackScholes, ThrowsOnNegativeSpot) {
    BSParams p = atm(); p.S = -1.0;
    EXPECT_THROW(BlackScholesEngine::price(p), std::invalid_argument);
}

TEST(BlackScholes, ThrowsOnZeroVol) {
    BSParams p = atm(); p.sigma = 0.0;
    EXPECT_THROW(BlackScholesEngine::price(p), std::invalid_argument);
}

TEST(BlackScholes, ThrowsOnNegativeExpiry) {
    BSParams p = atm(); p.T = -0.5;
    EXPECT_THROW(BlackScholesEngine::price(p), std::invalid_argument);
}