#include "black_scholes.h"

#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>
#include <numbers>

namespace pricing {

// ─────────────────────────────────────────────
//  Input validation
// ─────────────────────────────────────────────
void BSParams::validate() const {
    if (S     <= 0.0) throw std::invalid_argument("Spot price S must be > 0");
    if (K     <= 0.0) throw std::invalid_argument("Strike K must be > 0");
    if (sigma <= 0.0) throw std::invalid_argument("Volatility sigma must be > 0");
    if (T     <= 0.0) throw std::invalid_argument("Time to expiry T must be > 0");
    if (r < -1.0 || r > 10.0) throw std::invalid_argument("Rate r looks unreasonable");
    if (q <  0.0) throw std::invalid_argument("Dividend yield q must be >= 0");
}

// ─────────────────────────────────────────────
//  Normal distribution helpers
// ─────────────────────────────────────────────
double BlackScholesEngine::norm_pdf(double x) {
    return std::numbers::inv_sqrtpi / std::numbers::sqrt2 * std::exp(-0.5 * x * x);
}

double BlackScholesEngine::norm_cdf(double x) {
    constexpr double inv_sqrt2 = 1.0 / std::numbers::sqrt2;
    return 0.5 * std::erfc(-x * inv_sqrt2);
}

// ─────────────────────────────────────────────
//  d1 / d2 computation
// ─────────────────────────────────────────────
void BlackScholesEngine::computeD1D2(const BSParams& p, double& d1, double& d2) {
    const double sqrtT = std::sqrt(p.T);
    d1 = (std::log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma * p.sigma) * p.T)
         / (p.sigma * sqrtT);
    d2 = d1 - p.sigma * sqrtT;
}

// ─────────────────────────────────────────────
//  Price + full Greeks in one pass
// ─────────────────────────────────────────────
BSResult BlackScholesEngine::price(const BSParams& p) {
    p.validate();

    double d1, d2;
    computeD1D2(p, d1, d2);

    const double sqrtT     = std::sqrt(p.T);
    const double df        = std::exp(-p.r * p.T);    // discount factor
    const double df_q      = std::exp(-p.q * p.T);    // dividend discount
    const double nd1       = norm_cdf(d1);
    const double nd2       = norm_cdf(d2);
    const double nd1_neg   = norm_cdf(-d1);
    const double nd2_neg   = norm_cdf(-d2);
    const double pdf_d1    = norm_pdf(d1);

    BSResult result{};

    // ── Price ──
    result.price = p.type == OptionType::Call
        ? p.S * df_q * nd1 - p.K * df * nd2
        : p.K * df * nd2_neg - p.S * df_q * nd1_neg;

    Greeks& g = result.greeks; // alias for brevity

    // ── Delta ──
    g.delta = p.type == OptionType::Call
        ? df_q * nd1
        : df_q * (nd1 - 1.0);

    // ── Gamma (same for call & put) ──
    g.gamma = df_q * pdf_d1 / (p.S * p.sigma * sqrtT);

    // ── Theta (per calendar day) ──
    const double common_theta = -(p.S * df_q * pdf_d1 * p.sigma) / (2.0 * sqrtT);
    g.theta = p.type == OptionType::Call
        ? (common_theta - p.r * p.K * df * nd2 + p.q * p.S * df_q * nd1) / 365.0
        : (common_theta + p.r * p.K * df * nd2_neg - p.q * p.S * df_q * nd1_neg) / 365.0;


    // ── Vega (per 1% vol move) ──
    g.vega = p.S * df_q * pdf_d1 * sqrtT / 100.0;

    // ── Rho (per 1% rate move) ──
    g.rho = p.type == OptionType::Call
        ? p.K * p.T * df * nd2 / 100.0
        : -p.K * p.T * df * nd2_neg / 100.0;

    return result;
}

// ─────────────────────────────────────────────
//  Standalone Greek accessors
// ─────────────────────────────────────────────
double BlackScholesEngine::delta(const BSParams& p) { return price(p).greeks.delta; }
double BlackScholesEngine::gamma(const BSParams& p) { return price(p).greeks.gamma; }
double BlackScholesEngine::theta(const BSParams& p) { return price(p).greeks.theta; }
double BlackScholesEngine::vega (const BSParams& p) { return price(p).greeks.vega;  }
double BlackScholesEngine::rho  (const BSParams& p) { return price(p).greeks.rho;   }

// ─────────────────────────────────────────────
//  Implied Volatility — Newton-Raphson with Brent fallback
// ─────────────────────────────────────────────
double BlackScholesEngine::impliedVol(
    double marketPrice,
    double S, double K, double r, double q, double T,
    OptionType type,
    double tol,
    int    maxIter)
{
    // Intrinsic value lower bound check
    const double df   = std::exp(-r * T);
    const double df_q = std::exp(-q * T);
    const double intrinsic = type == OptionType::Call
        ? std::max(0.0, S * df_q - K * df)
        : std::max(0.0, K * df - S * df_q);

    if (marketPrice < intrinsic)
        throw std::runtime_error("Market price below intrinsic value — no valid IV exists");

    // Helper: price at given sigma
    auto priceAt = [&](double sigma) -> double {
        BSParams p{S, K, r, q, sigma, T, type};
        return price(p).price;
    };
    auto vegaAt = [&](double sigma) -> double {
        BSParams p{S, K, r, q, sigma, T, type};
        return price(p).greeks.vega * 100.0; // undo the /100 scaling
    };

    // ── Newton-Raphson phase ──
    double sigma = 0.2; // initial guess
    for (int i = 0; i < maxIter; ++i) {
        const double pv  = priceAt(sigma);
        const double err = pv - marketPrice;
        if (std::abs(err) < tol) return sigma;

        const double v = vegaAt(sigma);
        if (std::abs(v) < 1e-12) break; // flat vega — switch to Brent

        sigma -= err / v;
        if (sigma <= 0.0) sigma = 1e-6; // stay positive
    }

    // ── Brent fallback ──
    double lo = 1e-6, hi = 10.0;
    if ((priceAt(lo) - marketPrice) * (priceAt(hi) - marketPrice) > 0.0)
        throw std::runtime_error("IV solver: no root bracketed in [1e-6, 10.0]");

    for (int i = 0; i < maxIter; ++i) {
        const double mid = 0.5 * (lo + hi);
        const double err = priceAt(mid) - marketPrice;
        if (std::abs(err) < tol || (hi - lo) < tol) return mid;
        if (err > 0.0) hi = mid; else lo = mid;
    }

    throw std::runtime_error("IV solver failed to converge within " +
                             std::to_string(maxIter) + " iterations");
}

} // namespace pricing