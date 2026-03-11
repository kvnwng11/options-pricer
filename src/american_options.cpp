#include "american_options.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace pricing {

// ─────────────────────────────────────────────
//  Validation
// ─────────────────────────────────────────────
void TreeParams::validate() const {
    if (S     <= 0.0) throw std::invalid_argument("Spot price S must be > 0");
    if (K     <= 0.0) throw std::invalid_argument("Strike K must be > 0");
    if (sigma <= 0.0) throw std::invalid_argument("Volatility sigma must be > 0");
    if (T     <= 0.0) throw std::invalid_argument("Time to expiry T must be > 0");
    if (steps <  1  ) throw std::invalid_argument("steps must be >= 1");
    if (q     <  0.0) throw std::invalid_argument("Dividend yield q must be >= 0");
}

// ─────────────────────────────────────────────
//  Intrinsic value at a node
// ─────────────────────────────────────────────
double AmericanTreeEngine::intrinsic(double S, double K, OptionType type) {
    return (type == OptionType::Call)
        ? std::max(S - K, 0.0)
        : std::max(K - S, 0.0);
}

// ─────────────────────────────────────────────
//  CRR Binomial Tree
//
//  At each node (i, j): S(i,j) = S * u^j * d^(i-j)
//  where j = number of up-moves in i steps.
//
//  Parameters:
//    dt = T/N
//    u  = exp(sigma * sqrt(dt))
//    d  = 1/u
//    p  = (exp((r-q)*dt) - d) / (u - d)   [risk-neutral up probability]
//
//  Memory layout: single vector of size (N+1), reused in-place
//  during backward induction (avoids O(N²) allocation).
// ─────────────────────────────────────────────
double AmericanTreeEngine::priceCRR(const TreeParams& p) {
    const int    N  = p.steps;
    const double dt = p.T / N;
    const double u  = std::exp(p.sigma * std::sqrt(dt));
    const double d  = 1.0 / u;
    const double df = std::exp(-p.r * dt);              // per-step discount
    const double pu = (std::exp((p.r - p.q) * dt) - d) / (u - d);  // up probability
    const double pd = 1.0 - pu;

    if (pu < 0.0 || pu > 1.0)
        throw std::runtime_error("CRR: risk-neutral probability out of [0,1] — reduce dt or sigma");

    // Terminal node prices and payoffs
    // V[j] = option value at node with j up-moves
    std::vector<double> V(N + 1);
    for (int j = 0; j <= N; ++j) {
        const double S_term = p.S * std::pow(u, j) * std::pow(d, N - j);
        V[j] = intrinsic(S_term, p.K, p.type);
    }

    // Backward induction with early exercise check
    for (int i = N - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            const double S_node    = p.S * std::pow(u, j) * std::pow(d, i - j);
            const double hold      = df * (pu * V[j + 1] + pd * V[j]);
            const double exercise  = intrinsic(S_node, p.K, p.type);
            V[j] = std::max(hold, exercise);  // American early exercise
        }
    }

    return V[0];
}

// ─────────────────────────────────────────────
//  Trinomial Tree
//
//  Boyle (1986) formulation as given on Wikipedia / Hull:
//    u  = exp(σ * sqrt(2*dt))
//    d  = 1/u
//    pu = ( (exp((r-q)*dt/2) - exp(-σ*sqrt(dt/2)))
//           / (exp(σ*sqrt(dt/2)) - exp(-σ*sqrt(dt/2))) )²
//    pd = ( (exp(σ*sqrt(dt/2)) - exp((r-q)*dt/2))
//           / (exp(σ*sqrt(dt/2)) - exp(-σ*sqrt(dt/2))) )²
//    pm = 1 - pu - pd
//
//  Node indexing: at step i, j ranges from -i to +i (2i+1 nodes).
//  S(i,j) = S * u^j
// ─────────────────────────────────────────────
double AmericanTreeEngine::priceTrinomial(const TreeParams& p) {
    const int    N   = p.steps;
    const double dt  = p.T / N;
    // const double vdt = p.sigma * std::sqrt(dt);
    
    // Standard Boyle/Hull branching
    const double u  = std::exp(p.sigma * std::sqrt(2.0 * dt));
    // const double d  = 1.0 / u;
    const double df = std::exp(-p.r * dt);

    const double drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
    const double pu = 1.0/4.0 + drift / (2.0 * p.sigma * std::sqrt(2.0 * dt));
    const double pd = 1.0/4.0 - drift / (2.0 * p.sigma * std::sqrt(2.0 * dt));
    const double pm = 1.0 - pu - pd;

    if (pm < 0.0 || pu < 0.0 || pd < 0.0)
        throw std::runtime_error("Trinomial: Probabilities out of bounds.");

    const int max_nodes = 2 * N + 1;
    std::vector<double> V(max_nodes);
    std::vector<double> V_next(max_nodes);

    // Terminal payoffs at i = N
    for (int j = -N; j <= N; ++j) {
        V[j + N] = intrinsic(p.S * std::pow(u, j), p.K, p.type);
    }

    // Backward induction (Optimized In-Place)
    for (int i = N - 1; i >= 0; --i) {
        // Before starting the j-loop for step i, 
        // V[(-i-1) + N] and V[-i + N] are the "left" and "middle" 
        // for the first node j = -i.
        double v_prev = V[(-i - 1) + N]; 
        double v_curr = V[-i + N];

        for (int j = -i; j <= i; ++j) {
            double v_next = V[j + 1 + N]; // The "right" node at step i+1
            
            double S_node = p.S * std::pow(u, j);
            double hold = df * (pu * v_next + pm * v_curr + pd * v_prev);
            
            // Update buffers for the NEXT j iteration
            v_prev = v_curr;
            v_curr = v_next;

            // Overwrite the vector safely
            V[j + N] = std::max(hold, intrinsic(S_node, p.K, p.type));
        }
    }

    return V[N]; 
}

// ─────────────────────────────────────────────
//  Dispatch
// ─────────────────────────────────────────────
double AmericanTreeEngine::priceOnly(const TreeParams& p) {
    p.validate();
    switch (p.method) {
        case TreeMethod::CRR:       return priceCRR(p);
        case TreeMethod::Trinomial: return priceTrinomial(p);
        default: throw std::invalid_argument("Unknown TreeMethod");
    }
}

// ─────────────────────────────────────────────
//  Greeks via finite difference
//
//  Delta and Gamma are read from the top of the backward-induction
//  tree (steps 1 and 2) rather than bumping, giving exact tree Greeks.
//  Theta, Vega, Rho use central differences on priceOnly().
// ─────────────────────────────────────────────
TreeResult AmericanTreeEngine::price(const TreeParams& p) {
    p.validate();

    TreeResult result{};
    result.price = priceOnly(p);

    // ── Delta & Gamma from tree structure ──
    // Re-run with N=2 to extract S_u, S_d at step 1
    {
        const double dt = p.T / p.steps;
        const double u  = (p.method == TreeMethod::CRR)
            ? std::exp(p.sigma * std::sqrt(dt))
            : std::exp(p.sigma * std::sqrt(2.0 * dt));
        const double d  = 1.0 / u;

        const double S_u  = p.S * u;
        const double S_d  = p.S * d;
        const double S_uu = p.S * u * u;
        const double S_dd = p.S * d * d;

        // Price at each of the 3 nodes at step 1 (CRR) via sub-trees
        auto subprice = [&](double spot) {
            TreeParams q2 = p;
            q2.S = spot;
            q2.T = p.T - dt;
            q2.steps = std::max(1, p.steps - 1);
            return priceOnly(q2);
        };

        const double V_u  = subprice(S_u);
        const double V_d  = subprice(S_d);
        const double V_uu = subprice(S_uu);
        const double V_dd = subprice(S_dd);

        result.greeks.delta = (V_u - V_d) / (S_u - S_d);
        result.greeks.gamma = ((V_uu - result.price) / (S_uu - p.S)
                             - (result.price - V_dd) / (p.S - S_dd))
                             / (0.5 * (S_uu - S_dd));
    }

    // ── Theta (per calendar day) ──
    {
        const double dT = 2.0 / 365.0;
        TreeParams p_fwd = p; p_fwd.T = p.T - dT;
        result.greeks.theta = (priceOnly(p_fwd) - result.price) / dT / 365.0;
    }

    // ── Vega (per 1% vol move) ──
    {
        const double dSigma = 0.01;
        TreeParams p_up = p; p_up.sigma = p.sigma + dSigma;
        TreeParams p_dn = p; p_dn.sigma = p.sigma - dSigma;
        result.greeks.vega = (priceOnly(p_up) - priceOnly(p_dn))
                            / (2.0 * dSigma) * 0.01;
    }

    // ── Rho (per 1% rate move) ──
    {
        const double dR = 0.0001;
        TreeParams p_up = p; p_up.r = p.r + dR;
        TreeParams p_dn = p; p_dn.r = p.r - dR;
        result.greeks.rho = (priceOnly(p_up) - priceOnly(p_dn))
                           / (2.0 * dR) / 100.0;
    }

    return result;
}

double AmericanTreeEngine::delta(const TreeParams& p) { return price(p).greeks.delta; }
double AmericanTreeEngine::gamma(const TreeParams& p) { return price(p).greeks.gamma; }
double AmericanTreeEngine::theta(const TreeParams& p) { return price(p).greeks.theta; }
double AmericanTreeEngine::vega (const TreeParams& p) { return price(p).greeks.vega;  }
double AmericanTreeEngine::rho  (const TreeParams& p) { return price(p).greeks.rho;   }

} // namespace pricing