#include <gtest/gtest.h>
#include "black_scholes.h"

#include <cmath>

using namespace pricing;

// ─────────────────────────────────────────────
//  Test fixtures & helpers
// ─────────────────────────────────────────────
static constexpr double PRICE_TOL  = 1e-6;   // pricing precision
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
//  1. Put-Call Parity
// ─────────────────────────────────────────────
TEST(BlackScholes, PutCallParity) {
    const double S = 100, K = 100, r = 0.05, q = 0.02, sigma = 0.20, T = 1.0;

    BSParams call_p{S, K, r, q, sigma, T, OptionType::Call};
    BSParams put_p {S, K, r, q, sigma, T, OptionType::Put };

    const double C = BlackScholesEngine::price(call_p).price;
    const double P = BlackScholesEngine::price(put_p ).price;

    // C - P = S*e^{-qT} - K*e^{-rT}
    const double lhs = C - P;
    const double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);

    EXPECT_NEAR(lhs, rhs, PRICE_TOL);
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
//  6. Implied Volatility — round-trip
// ─────────────────────────────────────────────
TEST(BlackScholes, IVRoundTripCall) {
    auto p = atm();
    const double mkt_price = BlackScholesEngine::price(p).price;
    const double iv = BlackScholesEngine::impliedVol(
        mkt_price, p.S, p.K, p.r, p.q, p.T, OptionType::Call);
    EXPECT_NEAR(iv, p.sigma, IV_TOL);
}

TEST(BlackScholes, IVRoundTripPut) {
    auto p = atm(); p.type = OptionType::Put;
    const double mkt_price = BlackScholesEngine::price(p).price;
    const double iv = BlackScholesEngine::impliedVol(
        mkt_price, p.S, p.K, p.r, p.q, p.T, OptionType::Put);
    EXPECT_NEAR(iv, p.sigma, IV_TOL);
}

TEST(BlackScholes, IVRoundTripOTMCall) {
    BSParams p{100.0, 110.0, 0.05, 0.0, 0.25, 0.5, OptionType::Call};
    const double mkt_price = BlackScholesEngine::price(p).price;
    const double iv = BlackScholesEngine::impliedVol(
        mkt_price, p.S, p.K, p.r, p.q, p.T, OptionType::Call);
    EXPECT_NEAR(iv, p.sigma, IV_TOL);
}

TEST(BlackScholes, IVRoundTripHighVol) {
    BSParams p{100.0, 100.0, 0.05, 0.0, 0.80, 1.0, OptionType::Call};
    const double mkt_price = BlackScholesEngine::price(p).price;
    const double iv = BlackScholesEngine::impliedVol(
        mkt_price, p.S, p.K, p.r, p.q, p.T, OptionType::Call);
    EXPECT_NEAR(iv, p.sigma, IV_TOL);
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