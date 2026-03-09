#pragma once

#include <stdexcept>
#include <string>

namespace pricing {

// ─────────────────────────────────────────────
//  Option type
// ─────────────────────────────────────────────
enum class OptionType { Call, Put };

// ─────────────────────────────────────────────
//  Market & contract inputs
// ─────────────────────────────────────────────
struct BSParams {
    double S;     // Spot price
    double K;     // Strike price
    double r;     // Risk-free rate (annualised, continuous)
    double q;     // Dividend yield (annualised, continuous)
    double sigma; // Implied volatility (annualised)
    double T;     // Time to expiry (years)
    OptionType type;

    void validate() const;
};

// ─────────────────────────────────────────────
//  Greeks bundle
// ─────────────────────────────────────────────
struct Greeks {
    double delta;  // ∂V/∂S
    double gamma;  // ∂²V/∂S²
    double theta;  // ∂V/∂t  (per calendar day)
    double vega;   // ∂V/∂σ  (per 1% move in vol)
    double rho;    // ∂V/∂r  (per 1% move in rate)
};

// ─────────────────────────────────────────────
//  Pricing result
// ─────────────────────────────────────────────
struct BSResult {
    double price;
    Greeks greeks;
};

// ─────────────────────────────────────────────
//  Black-Scholes Engine
// ─────────────────────────────────────────────
class BlackScholesEngine {
public:
    // Price an option and compute all Greeks in one pass
    static BSResult price(const BSParams& p);

    // Standalone Greek accessors (useful for bumping/scenario runs)
    static double delta(const BSParams& p);
    static double gamma(const BSParams& p);
    static double theta(const BSParams& p);
    static double vega (const BSParams& p);
    static double rho  (const BSParams& p);

    // Implied volatility solver (Newton-Raphson with Brent fallback)
    // Throws std::runtime_error if IV cannot be found within tolerance
    static double impliedVol(
        double marketPrice,
        double S, double K, double r, double q, double T,
        OptionType type,
        double tol     = 1e-8,
        int    maxIter = 200
    );

private:
    // Compute d1 and d2
    static void computeD1D2(const BSParams& p, double& d1, double& d2);

    // Standard normal PDF and CDF
    static double norm_pdf(double x);
    static double norm_cdf(double x);
};

} // namespace pricing