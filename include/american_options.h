#pragma once

#include "black_scholes.h"  // reuse OptionType, Greeks, BSParams

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace pricing {

// ─────────────────────────────────────────────
//  Tree method selector
// ─────────────────────────────────────────────
enum class TreeMethod {
    CRR,        // Cox-Ross-Rubinstein binomial tree
    Trinomial,  // Standard trinomial tree (better convergence)
};

// ─────────────────────────────────────────────
//  Tree engine parameters
// ─────────────────────────────────────────────
struct TreeParams {
    double S;           // Spot price
    double K;           // Strike price
    double r;           // Risk-free rate (annualised, continuous)
    double q;           // Dividend yield (annualised, continuous)
    double sigma;       // Volatility (annualised)
    double T;           // Time to expiry (years)
    OptionType type;    // Call or Put
    int    steps;       // Number of time steps (higher = more accurate, ~1000 typical)
    TreeMethod method;  // CRR or Trinomial

    void validate() const;
};

// ─────────────────────────────────────────────
//  Pricing result (price + numerical Greeks)
// ─────────────────────────────────────────────
struct TreeResult {
    double price;
    Greeks greeks;  // computed via finite difference on the tree
};

// ─────────────────────────────────────────────
//  American Options Tree Engine
//
//  Prices American options (with early exercise) using:
//    - Cox-Ross-Rubinstein (CRR) binomial tree
//    - Trinomial tree (improved convergence vs. CRR)
//
//  Greeks are computed by re-pricing with bumped parameters
//  (finite difference). Delta and Gamma are read directly from
//  the first two levels of the backward induction tree.
// ─────────────────────────────────────────────
class AmericanTreeEngine {
public:
    // Price and compute all Greeks
    static TreeResult price(const TreeParams& p);

    // Standalone accessors — each re-runs the tree with a bump
    static double delta(const TreeParams& p);
    static double gamma(const TreeParams& p);
    static double theta(const TreeParams& p);
    static double vega (const TreeParams& p);
    static double rho  (const TreeParams& p);

    // Price-only (no Greeks) — faster, used internally for bumping
    static double priceOnly(const TreeParams& p);

private:
    static double priceCRR       (const TreeParams& p);
    static double priceTrinomial  (const TreeParams& p);

    // Intrinsic value at a node
    static double intrinsic(double S, double K, OptionType type);
};

} // namespace pricing