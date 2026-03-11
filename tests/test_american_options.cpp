#include <gtest/gtest.h>
#include "american_options.h"
#include "black_scholes.h"

#include <cmath>
#include <cstdint>
#include <string>

using namespace pricing;

// ─────────────────────────────────────────────
//  Shared RNG — Xorshift64, deterministic & reproducible
// ─────────────────────────────────────────────
struct Rng {
    uint64_t seed;
    explicit Rng(uint64_t s) : seed(s) {}

    double next() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return static_cast<double>(seed >> 11) / static_cast<double>(1ULL << 53);
    }

    double uniform(double lo, double hi) { return lo + next() * (hi - lo); }
};

// ─────────────────────────────────────────────
//  Tolerances
// ─────────────────────────────────────────────
static constexpr double CONV_TOL  = 1e-2;   // tree vs BS (1 cent)
static constexpr double GREEK_TOL = 1e-2;   // tree Greek tolerance

// Canonical ATM params
static TreeParams atm(TreeMethod method = TreeMethod::CRR, int steps = 1000) {
    return {100.0, 100.0, 0.05, 0.02, 0.20, 1.0, OptionType::Put, steps, method};
}

// ─────────────────────────────────────────────
//  1. American call (q=0) == European BS call  (property-based)
//
//  Early exercise of an American call on a non-dividend-paying stock
//  is never optimal, so its price must equal the European BS price.
//  Tested across 200 randomised (S, K, r, sigma, T) combinations for
//  both CRR and Trinomial.
// ─────────────────────────────────────────────
TEST(AmericanTree, AmericanCallEqualsEuropeanCallNoDividend_PropertyBased) {
    Rng rng(0xABCDEF1234567890ULL);
    constexpr int N_TRIALS = 200;
    int failures = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = rng.uniform(20.0,  200.0);
        const double K     = rng.uniform(20.0,  200.0);
        const double r     = rng.uniform(0.01,  0.10);
        const double sigma = rng.uniform(0.10,  0.60);
        const double T     = rng.uniform(0.25,  2.0);
        const TreeMethod method = (i % 2 == 0) ? TreeMethod::CRR : TreeMethod::Trinomial;
        const int steps = (method == TreeMethod::CRR) ? 1000 : 500;

        TreeParams tp{S, K, r, 0.0, sigma, T, OptionType::Call, steps, method};
        BSParams   bp{S, K, r, 0.0, sigma, T, OptionType::Call};

        const double american = AmericanTreeEngine::priceOnly(tp);
        const double european = BlackScholesEngine::price(bp).price;

        if (std::abs(american - european) > CONV_TOL) {
            ADD_FAILURE() << "Trial " << i
                          << " method=" << (method == TreeMethod::CRR ? "CRR" : "Trinomial")
                          << " S=" << S << " K=" << K << " r=" << r
                          << " sigma=" << sigma << " T=" << T
                          << "\n  american=" << american
                          << " european=" << european
                          << " diff=" << std::abs(american - european);
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed";
}

// ─────────────────────────────────────────────
//  2. American >= European  (property-based)
//
//  An American option always has at least as much value as its
//  European counterpart (the early exercise right has non-negative value).
//  Tested for both puts (always valuable) and calls with dividends.
// ─────────────────────────────────────────────
TEST(AmericanTree, AmericanGeqEuropean_PropertyBased) {
    Rng rng(0x1234ABCD5678EF90ULL);
    constexpr int N_TRIALS = 200;
    int failures = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = rng.uniform(20.0,  200.0);
        const double K     = rng.uniform(20.0,  200.0);
        const double r     = rng.uniform(0.01,  0.10);
        const double q     = rng.uniform(0.0,   0.08);
        const double sigma = rng.uniform(0.10,  0.60);
        const double T     = rng.uniform(0.25,  2.0);
        const OptionType type = (i % 2 == 0) ? OptionType::Put : OptionType::Call;

        TreeParams tp{S, K, r, q, sigma, T, type, 500, TreeMethod::CRR};
        BSParams   bp{S, K, r, q, sigma, T, type};

        const double american = AmericanTreeEngine::priceOnly(tp);
        const double european = BlackScholesEngine::price(bp).price;

        if (american < european - CONV_TOL) {
            ADD_FAILURE() << "Trial " << i
                          << " type=" << (type == OptionType::Put ? "Put" : "Call")
                          << " S=" << S << " K=" << K << " r=" << r << " q=" << q
                          << " sigma=" << sigma << " T=" << T
                          << "\n  american=" << american << " european=" << european;
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed";
}

// ─────────────────────────────────────────────
//  3. Intrinsic value lower bound  (property-based)
//
//  An American option can always be exercised immediately, so its
//  price must be >= max(S-K, 0) for calls, max(K-S, 0) for puts.
// ─────────────────────────────────────────────
TEST(AmericanTree, PriceGeqIntrinsic_PropertyBased) {
    Rng rng(0xFEEDFACEDEADBEEFULL);
    constexpr int N_TRIALS = 300;
    int failures = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = rng.uniform(10.0,  300.0);
        const double K     = rng.uniform(10.0,  300.0);
        const double r     = rng.uniform(0.0,   0.10);
        const double q     = rng.uniform(0.0,   0.08);
        const double sigma = rng.uniform(0.05,  0.80);
        const double T     = rng.uniform(1.0/52.0, 3.0);
        const OptionType type = (i % 2 == 0) ? OptionType::Put : OptionType::Call;

        TreeParams tp{S, K, r, q, sigma, T, type, 200, TreeMethod::CRR};
        const double price_    = AmericanTreeEngine::priceOnly(tp);
        const double intrinsic = (type == OptionType::Call)
            ? std::max(S - K, 0.0)
            : std::max(K - S, 0.0);

        if (price_ < intrinsic - 1e-6) {
            ADD_FAILURE() << "Trial " << i
                          << " type=" << (type == OptionType::Put ? "Put" : "Call")
                          << " S=" << S << " K=" << K
                          << "\n  price=" << price_ << " intrinsic=" << intrinsic;
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed";
}

// ─────────────────────────────────────────────
//  4. CRR and Trinomial agree  (property-based)
//
//  Both methods must converge to the same price. At sufficient step
//  counts the difference should be < 1 cent.
// ─────────────────────────────────────────────
TEST(AmericanTree, CRRAndTrinomialAgree_PropertyBased) {
    Rng rng(0xCAFEBABE12345678ULL);
    constexpr int N_TRIALS = 100;
    int failures = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = rng.uniform(40.0,  160.0);
        const double K     = rng.uniform(40.0,  160.0);
        const double r     = rng.uniform(0.01,  0.08);
        const double q     = rng.uniform(0.0,   0.05);
        const double sigma = rng.uniform(0.10,  0.50);
        const double T     = rng.uniform(0.25,  2.0);
        const OptionType type = (i % 2 == 0) ? OptionType::Put : OptionType::Call;

        TreeParams crr{S, K, r, q, sigma, T, type, 1000, TreeMethod::CRR};
        TreeParams tri{S, K, r, q, sigma, T, type,  500, TreeMethod::Trinomial};

        const double v_crr = AmericanTreeEngine::priceOnly(crr);
        const double v_tri = AmericanTreeEngine::priceOnly(tri);

        if (std::abs(v_crr - v_tri) > CONV_TOL) {
            ADD_FAILURE() << "Trial " << i
                          << " type=" << (type == OptionType::Put ? "Put" : "Call")
                          << " S=" << S << " K=" << K << " r=" << r << " q=" << q
                          << " sigma=" << sigma << " T=" << T
                          << "\n  CRR=" << v_crr << " Tri=" << v_tri
                          << " diff=" << std::abs(v_crr - v_tri);
            ++failures;
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed";
}

// ─────────────────────────────────────────────
//  5. Monotone convergence in N
// ─────────────────────────────────────────────
TEST(AmericanTree, PriceConvergesAsStepsIncrease) {
    auto p100  = atm(TreeMethod::CRR, 100);
    auto p500  = atm(TreeMethod::CRR, 500);
    auto p1000 = atm(TreeMethod::CRR, 1000);

    const double v100  = AmericanTreeEngine::priceOnly(p100);
    const double v500  = AmericanTreeEngine::priceOnly(p500);
    const double v1000 = AmericanTreeEngine::priceOnly(p1000);

    EXPECT_LT(std::abs(v1000 - v500), std::abs(v500 - v100));
}

// ─────────────────────────────────────────────
//  6. Greeks — sign checks  (property-based)
//
//  Standard Greek sign conventions must hold across a range of
//  inputs for both puts and calls.
// ─────────────────────────────────────────────
TEST(AmericanTree, GreekSigns_PropertyBased) {
    Rng rng(0x0F0F0F0F0F0F0F0FULL);
    constexpr int N_TRIALS = 40;  // Greeks are expensive (multiple re-pricings each)
    int failures = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = rng.uniform(50.0,  150.0);
        const double K     = rng.uniform(50.0,  150.0);
        const double r     = rng.uniform(0.01,  0.08);
        const double q     = rng.uniform(0.0,   0.05);
        const double sigma = rng.uniform(0.10,  0.50);
        const double T     = rng.uniform(0.25,  2.0);
        const OptionType type = (i % 2 == 0) ? OptionType::Put : OptionType::Call;

        // Skip deep ITM: Greek sign conventions only hold where optionality exists.
        // Deep ITM options correctly have delta = +/-1 and all other Greeks = 0.
        const double moneyness = S / K;
        const bool deep_itm = (type == OptionType::Call && moneyness > 1.5)
                           || (type == OptionType::Put  && moneyness < 0.67);
        if (deep_itm) continue;

        TreeParams tp{S, K, r, q, sigma, T, type, 200, TreeMethod::CRR};
        const auto result = AmericanTreeEngine::price(tp);
        const auto& g = result.greeks;

        auto fail = [&](const char* msg) {
            ADD_FAILURE() << "Trial " << i << " " << msg
                          << " type=" << (type == OptionType::Put ? "Put" : "Call")
                          << " S=" << S << " K=" << K
                          << " delta=" << g.delta << " gamma=" << g.gamma
                          << " theta=" << g.theta << " vega=" << g.vega
                          << " rho=" << g.rho;
            ++failures;
        };

        // Use a small negative tolerance rather than strict > 0, so that
        // legitimate near-zero Greeks (e.g. far OTM) don't false-positive.
        constexpr double G_TOL = -1e-9;
        if (g.gamma  <  G_TOL)                               fail("gamma < 0");
        if (g.vega   <  G_TOL)                               fail("vega < 0");
        if (g.theta  > -G_TOL)                               fail("theta > 0");
        if (type == OptionType::Call && g.delta <  G_TOL)    fail("call delta < 0");
        if (type == OptionType::Call && g.delta >  1.0+1e-9) fail("call delta > 1");
        if (type == OptionType::Put  && g.delta > -G_TOL)    fail("put delta > 0");
        if (type == OptionType::Put  && g.delta < -1.0-1e-9) fail("put delta < -1");
        if (type == OptionType::Call && g.rho   <  G_TOL)    fail("call rho < 0");
        if (type == OptionType::Put  && g.rho   > -G_TOL)    fail("put rho > 0");
    }
    EXPECT_EQ(failures, 0) << failures << " Greek sign violations";
}

// ─────────────────────────────────────────────
//  7. Greeks match Black-Scholes for American call (q=0)  (property-based)
//
//  When early exercise is never optimal (call, q=0), American and
//  European Greeks must agree within tree discretisation error.
// ─────────────────────────────────────────────
TEST(AmericanTree, GreeksMatchBS_CallNoDividend_PropertyBased) {
    Rng rng(0xDEADC0DEBEEF1234ULL);
    constexpr int N_TRIALS = 30;
    int failures = 0;

    for (int i = 0; i < N_TRIALS; ++i) {
        const double S     = rng.uniform(50.0,  150.0);
        const double K     = rng.uniform(50.0,  150.0);
        const double r     = rng.uniform(0.01,  0.08);
        const double sigma = rng.uniform(0.10,  0.50);
        const double T     = rng.uniform(0.5,   2.0);

        TreeParams tp{S, K, r, 0.0, sigma, T, OptionType::Call, 500, TreeMethod::CRR};
        BSParams   bp{S, K, r, 0.0, sigma, T, OptionType::Call};

        const auto tree_g = AmericanTreeEngine::price(tp).greeks;
        const auto bs_g   = BlackScholesEngine::price(bp).greeks;

        auto check = [&](const char* name, double tv, double bv) {
            if (std::abs(tv - bv) > GREEK_TOL) {
                ADD_FAILURE() << "Trial " << i << " " << name
                              << ": tree=" << tv << " BS=" << bv
                              << " diff=" << std::abs(tv - bv)
                              << " S=" << S << " K=" << K
                              << " sigma=" << sigma << " T=" << T;
                ++failures;
            }
        };

        check("delta", tree_g.delta, bs_g.delta);
        check("vega",  tree_g.vega,  bs_g.vega);
        check("rho",   tree_g.rho,   bs_g.rho);
    }
    EXPECT_EQ(failures, 0) << failures << "/" << N_TRIALS << " trials failed";
}

// ─────────────────────────────────────────────
//  8. Input validation
// ─────────────────────────────────────────────
TEST(AmericanTree, ThrowsOnNegativeSpot) {
    TreeParams tp = atm(); tp.S = -1.0;
    EXPECT_THROW(AmericanTreeEngine::priceOnly(tp), std::invalid_argument);
}

TEST(AmericanTree, ThrowsOnZeroSteps) {
    TreeParams tp = atm(); tp.steps = 0;
    EXPECT_THROW(AmericanTreeEngine::priceOnly(tp), std::invalid_argument);
}

TEST(AmericanTree, ThrowsOnZeroVol) {
    TreeParams tp = atm(); tp.sigma = 0.0;
    EXPECT_THROW(AmericanTreeEngine::priceOnly(tp), std::invalid_argument);
}