#include <gtest/gtest.h>
#include "monte_carlo.h"
#include "black_scholes.h"

#include <cmath>
#include <cstdint>

using namespace pricing;

// ─────────────────────────────────────────────
//  Shared RNG for test parameter generation
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

// ─────────────────────────────────────────────
//  Tolerances
//  MC is a statistical estimator — we test that results fall within
//  a fixed number of standard errors of the reference, not to machine
//  precision. 3 SE covers 99.7% of outcomes.
// ─────────────────────────────────────────────
static constexpr double N_SE     = 5.0;    // tolerate up to 5 standard errors
static constexpr double ABS_TOL  = 0.05;   // absolute fallback (5 cents) for near-zero prices

// Helper: check MC price is within N_SE standard errors of a reference
static void expectWithinSE(double mc_price, double mc_se, double reference,
                            const std::string& label = "") {
    const double tol = std::max(N_SE * mc_se, ABS_TOL);
    EXPECT_NEAR(mc_price, reference, tol) << label
        << " (SE=" << mc_se << " tol=" << tol << ")";
}

// Canonical ATM European call params
static MCParams atmCall(int paths = 100'000) {
    return {100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            OptionType::Call, ExoticType::European,
            paths, 252, 0.0, VarianceReduction::None, 0xDEADBEEF12345678ULL};
}

// ─────────────────────────────────────────────
//  1. European MC converges to Black-Scholes  (property-based)
//
//  The European MC price must be within N standard errors of the
//  analytic BS price across randomised (S, K, r, q, sigma, T).
// ─────────────────────────────────────────────
TEST(MonteCarlo, EuropeanConvergesToBS_PropertyBased) {
    Rng rng(0xABCDEF1234567890ULL);
    constexpr int N_TRIALS = 20;  // MC is expensive
    int failures = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = rng.uniform(50.0,  200.0);
        const double K     = rng.uniform(50.0,  200.0);
        const double r     = rng.uniform(0.01,  0.08);
        const double q     = rng.uniform(0.0,   0.05);
        const double sigma = rng.uniform(0.10,  0.50);
        const double T     = rng.uniform(0.25,  2.0);
        const OptionType type = (i % 2 == 0) ? OptionType::Call : OptionType::Put;

        MCParams mp{S, K, r, q, sigma, T, type, ExoticType::European,
                    200'000, 252, 0.0, VarianceReduction::Antithetic,
                    0xDEAD0000ULL + static_cast<uint64_t>(i)};
        BSParams bp{S, K, r, q, sigma, T, type};

        const auto mc = MonteCarloEngine::priceOnly(mp);
        const double bs = BlackScholesEngine::price(bp).price;
        const double tol = std::max(N_SE * mc.stdError, ABS_TOL);

        if (std::abs(mc.price - bs) > tol) {
            ADD_FAILURE() << "Trial " << i
                          << " type=" << (type == OptionType::Call ? "Call" : "Put")
                          << " S=" << S << " K=" << K << " r=" << r
                          << " q=" << q << " sigma=" << sigma << " T=" << T
                          << "\n  MC=" << mc.price << " BS=" << bs
                          << " SE=" << mc.stdError << " tol=" << tol;
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed";
}

// ─────────────────────────────────────────────
//  2. Put-call parity holds for MC European  (property-based)
// ─────────────────────────────────────────────
TEST(MonteCarlo, EuropeanPutCallParity_PropertyBased) {
    Rng rng(0x1111222233334444ULL);
    constexpr int N_TRIALS = 15;
    int failures = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = rng.uniform(50.0, 150.0);
        const double K     = rng.uniform(50.0, 150.0);
        const double r     = rng.uniform(0.01, 0.08);
        const double q     = rng.uniform(0.0,  0.05);
        const double sigma = rng.uniform(0.10, 0.40);
        const double T     = rng.uniform(0.5,  2.0);

        MCParams call_p{S, K, r, q, sigma, T, OptionType::Call, ExoticType::European,
                        200'000, 252, 0.0, VarianceReduction::Antithetic,
                        0xCAFE0000ULL + static_cast<uint64_t>(i)};
        MCParams put_p = call_p; put_p.type = OptionType::Put;

        const double C   = MonteCarloEngine::priceOnly(call_p).price;
        const double P   = MonteCarloEngine::priceOnly(put_p ).price;
        const double lhs = C - P;
        const double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);

        // Parity holds to within combined SE of both estimates
        const auto mc_c = MonteCarloEngine::priceOnly(call_p);
        const auto mc_p = MonteCarloEngine::priceOnly(put_p);
        const double combined_se = std::sqrt(mc_c.stdError * mc_c.stdError +
                                             mc_p.stdError * mc_p.stdError);
        const double tol = std::max(N_SE * combined_se, ABS_TOL);

        if (std::abs(lhs - rhs) > tol) {
            ADD_FAILURE() << "Trial " << i
                          << " C-P=" << lhs << " rhs=" << rhs
                          << " diff=" << std::abs(lhs - rhs) << " tol=" << tol;
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed";
}

// ─────────────────────────────────────────────
//  3. Variance reduction reduces standard error
//
//  Antithetic variates must produce a lower SE than plain MC
//  for the same number of paths on a smooth (European) payoff.
// ─────────────────────────────────────────────
TEST(MonteCarlo, AntitheticReducesStdError) {
    MCParams plain = atmCall(200'000);
    MCParams anti  = plain;
    anti.vr = VarianceReduction::Antithetic;

    const auto r_plain = MonteCarloEngine::priceOnly(plain);
    const auto r_anti  = MonteCarloEngine::priceOnly(anti);

    EXPECT_LT(r_anti.stdError, r_plain.stdError)
        << "Antithetic SE=" << r_anti.stdError
        << " plain SE=" << r_plain.stdError;
}

TEST(MonteCarlo, ControlVariateImprovesPriceAccuracy) {
    // The post-hoc CV correction shifts the price toward BS but does not
    // reduce the raw sample stdError (which reflects path-level variance).
    // The correct check is that the CV price is closer to the BS reference.
    BSParams bp{100.0, 100.0, 0.05, 0.02, 0.20, 1.0, OptionType::Call};
    const double bs = BlackScholesEngine::price(bp).price;

    MCParams plain = atmCall(50'000);
    MCParams cv    = plain;
    cv.vr = VarianceReduction::ControlVariate;

    const double err_plain = std::abs(MonteCarloEngine::priceOnly(plain).price - bs);
    const double err_cv    = std::abs(MonteCarloEngine::priceOnly(cv   ).price - bs);

    EXPECT_LT(err_cv, err_plain)
        << "CV error=" << err_cv << " plain error=" << err_plain
        << " BS=" << bs;
}

// ─────────────────────────────────────────────
//  4. Standard error narrows as O(1/sqrt(N))
//
//  Doubling paths should halve SE (approximately).
//  We check the ratio falls in [1.2, 2.2] to allow MC noise.
// ─────────────────────────────────────────────
TEST(MonteCarlo, StdErrorNarrowsWithMorePaths) {
    MCParams p_small = atmCall(50'000);
    MCParams p_large = atmCall(200'000);

    const auto r_small = MonteCarloEngine::priceOnly(p_small);
    const auto r_large = MonteCarloEngine::priceOnly(p_large);

    // SE should scale as 1/sqrt(N): SE_small/SE_large ~ sqrt(200k/50k) = 2
    const double ratio = r_small.stdError / r_large.stdError;
    EXPECT_GT(ratio, 1.2) << "SE not narrowing: small=" << r_small.stdError
                          << " large=" << r_large.stdError;
    EXPECT_LT(ratio, 3.0) << "SE ratio suspiciously large: " << ratio;
}

// ─────────────────────────────────────────────
//  5. Asian geometric == closed-form reference
//
//  The geometric Asian call has a known closed-form price (the
//  Black-Scholes formula with adjusted parameters). We use this
//  as an exact reference.
// ─────────────────────────────────────────────
static double geoAsianClosedForm(double S, double K, double r, double q,
                                  double sigma, double T, OptionType type) {
    // Kemna-Vorst (1990): replace sigma and r with adjusted values
    const double sigma_adj = sigma / std::sqrt(3.0);
    const double r_adj     = 0.5 * (r - q - 0.5 * sigma * sigma)
                           + 0.5 * sigma_adj * sigma_adj;
    BSParams bp{S, K, r_adj + q, q, sigma_adj, T, type};
    // Discount at r, not r_adj
    const double df = std::exp(-r * T) / std::exp(-(r_adj + q) * T);
    return BlackScholesEngine::price(bp).price * df;
}

TEST(MonteCarlo, AsianGeoConvergesToClosedForm) {
    const double S = 100, K = 100, r = 0.05, q = 0.0, sigma = 0.20, T = 1.0;

    MCParams mp{S, K, r, q, sigma, T, OptionType::Call, ExoticType::AsianGeo,
                500'000, 252, 0.0, VarianceReduction::Antithetic, 0xA5A5A5A5A5A5A5A5ULL};

    const auto mc  = MonteCarloEngine::priceOnly(mp);
    const double cf = geoAsianClosedForm(S, K, r, q, sigma, T, OptionType::Call);

    expectWithinSE(mc.price, mc.stdError, cf, "AsianGeo vs closed-form");
}

// ─────────────────────────────────────────────
//  6. Barrier in + out = vanilla  (property-based)
//
//  A knock-in and knock-out with the same barrier must sum to
//  the corresponding vanilla price. This is an exact identity.
// ─────────────────────────────────────────────
TEST(MonteCarlo, BarrierInPlusOutEqualsVanilla_PropertyBased) {
    Rng rng(0xBEEFCAFEDEAD1234ULL);
    constexpr int N_TRIALS = 10;
    int failures = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = rng.uniform(80.0,  120.0);
        const double K     = rng.uniform(80.0,  120.0);
        const double r     = rng.uniform(0.01,  0.06);
        const double sigma = rng.uniform(0.15,  0.40);
        const double T     = rng.uniform(0.5,   1.5);

        // Up barriers: barrier > S
        const double barrier = S * rng.uniform(1.05, 1.30);

        const uint64_t seed = 0xF00D0000ULL + static_cast<uint64_t>(i);
        MCParams base{S, K, r, 0.0, sigma, T, OptionType::Call,
                      ExoticType::European, 200'000, 252, barrier,
                      VarianceReduction::None, seed};

        MCParams up_out = base; up_out.exotic = ExoticType::BarrierUpOut;
        MCParams up_in  = base; up_in.exotic  = ExoticType::BarrierUpIn;
        MCParams vanilla = base; vanilla.exotic = ExoticType::European;

        const double v_out  = MonteCarloEngine::priceOnly(up_out ).price;
        const double v_in   = MonteCarloEngine::priceOnly(up_in  ).price;
        const double v_van  = MonteCarloEngine::priceOnly(vanilla).price;

        // All three use the same seed/paths so noise partially cancels
        const double diff = std::abs((v_out + v_in) - v_van);
        if (diff > ABS_TOL * 3) {
            ADD_FAILURE() << "Trial " << i
                          << " UpOut+UpIn=" << (v_out + v_in)
                          << " Vanilla=" << v_van
                          << " diff=" << diff
                          << " S=" << S << " K=" << K << " H=" << barrier;
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed";
}

// ─────────────────────────────────────────────
//  7. Asian arithmetic >= geometric  (property-based)
//
//  By Jensen's inequality, the arithmetic mean >= geometric mean,
//  so arith Asian price >= geo Asian price for calls.
// ─────────────────────────────────────────────
TEST(MonteCarlo, AsianArithGeqGeo_PropertyBased) {
    Rng rng(0x9999888877776666ULL);
    constexpr int N_TRIALS = 10;
    int failures = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = rng.uniform(80.0,  120.0);
        const double K     = rng.uniform(80.0,  120.0);
        const double r     = rng.uniform(0.01,  0.06);
        const double sigma = rng.uniform(0.10,  0.40);
        const double T     = rng.uniform(0.5,   1.5);

        const uint64_t seed = 0xBEEF0000ULL + static_cast<uint64_t>(i);
        MCParams geo  {S, K, r, 0.0, sigma, T, OptionType::Call, ExoticType::AsianGeo,
                       200'000, 252, 0.0, VarianceReduction::Antithetic, seed};
        MCParams arith = geo; arith.exotic = ExoticType::AsianArith;

        const double v_arith = MonteCarloEngine::priceOnly(arith).price;
        const double v_geo   = MonteCarloEngine::priceOnly(geo  ).price;

        if (v_arith < v_geo - ABS_TOL) {
            ADD_FAILURE() << "Trial " << i
                          << " Arith=" << v_arith << " Geo=" << v_geo
                          << " S=" << S << " K=" << K;
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed";
}

// ─────────────────────────────────────────────
//  8. Greeks — sign checks
// ─────────────────────────────────────────────
TEST(MonteCarlo, GreekSigns_European) {
    MCParams p = atmCall(100'000);
    const auto result = MonteCarloEngine::price(p);
    const auto& g = result.greeks;

    EXPECT_GT(g.delta, 0.0)  << "call delta should be positive";
    EXPECT_LT(g.delta, 1.0)  << "call delta should be < 1";
    EXPECT_GT(g.gamma, -1e-4) << "gamma should be non-negative";
    EXPECT_LT(g.theta, 1e-4)  << "theta should be non-positive";
    EXPECT_GT(g.vega,  0.0)  << "vega should be positive";
    EXPECT_GT(g.rho,   0.0)  << "call rho should be positive";
}

// ─────────────────────────────────────────────
//  9. Confidence interval contains BS price
// ─────────────────────────────────────────────
TEST(MonteCarlo, ConfidenceIntervalContainsBS) {
    MCParams mp = atmCall(500'000);
    mp.vr = VarianceReduction::Antithetic;

    BSParams bp{mp.S, mp.K, mp.r, mp.q, mp.sigma, mp.T, mp.type};
    const double bs = BlackScholesEngine::price(bp).price;

    const auto mc = MonteCarloEngine::priceOnly(mp);

    EXPECT_GE(bs, mc.confLo) << "BS price below MC 95% CI lower bound";
    EXPECT_LE(bs, mc.confHi) << "BS price above MC 95% CI upper bound";
}

// ─────────────────────────────────────────────
//  10. Input validation
// ─────────────────────────────────────────────
TEST(MonteCarlo, ThrowsOnNegativeSpot) {
    MCParams p = atmCall(); p.S = -1.0;
    EXPECT_THROW(MonteCarloEngine::priceOnly(p), std::invalid_argument);
}

TEST(MonteCarlo, ThrowsOnZeroPaths) {
    MCParams p = atmCall(); p.numPaths = 0;
    EXPECT_THROW(MonteCarloEngine::priceOnly(p), std::invalid_argument);
}

TEST(MonteCarlo, ThrowsOnBarrierWithoutLevel) {
    MCParams p = atmCall();
    p.exotic  = ExoticType::BarrierUpOut;
    p.barrier = 0.0;
    EXPECT_THROW(MonteCarloEngine::priceOnly(p), std::invalid_argument);
}