#pragma once

#include "black_scholes.h"  // reuse OptionType, Greeks

#include <cstdint>
#include <functional>
#include <stdexcept>

namespace pricing {

// ─────────────────────────────────────────────
//  Supported exotic payoff types
// ─────────────────────────────────────────────
enum class ExoticType {
    European,       // Standard call/put — baseline for validation
    AsianArith,     // Asian: arithmetic average of path vs strike
    AsianGeo,       // Asian: geometric average of path vs strike
    BarrierUpOut,   // Up-and-out: knocked out if S ever crosses barrier H from below
    BarrierUpIn,    // Up-and-in: activated only if S ever crosses barrier H from below
    BarrierDownOut, // Down-and-out: knocked out if S ever crosses barrier H from above
    BarrierDownIn,  // Down-and-in: activated only if S ever crosses barrier H from above
    Lookback,       // Lookback fixed-strike: payoff on max(put) or min(call) of path
};

// ─────────────────────────────────────────────
//  Variance reduction techniques
// ─────────────────────────────────────────────
enum class VarianceReduction {
    None,
    Antithetic,            // Antithetic variates: average path with its mirror
    ControlVariate,        // Control variate: use BS European as control
    AntitheticAndControl,  // Both combined
};

// ─────────────────────────────────────────────
//  Monte Carlo parameters
// ─────────────────────────────────────────────
struct MCParams {
    double S;           // Spot price
    double K;           // Strike price
    double r;           // Risk-free rate (annualised, continuous)
    double q;           // Dividend yield (annualised, continuous)
    double sigma;       // Volatility (annualised)
    double T;           // Time to expiry (years)
    OptionType   type;
    ExoticType   exotic;
    int          numPaths;   // Number of simulation paths (e.g. 1'000'000)
    int          numSteps;   // Time steps per path (e.g. 252 for daily)
    double       barrier;    // Barrier level H (only used for barrier options)
    VarianceReduction vr;    // Variance reduction method
    uint64_t     seed;       // Base RNG seed (per-thread seeds are derived from this)

    void validate() const;
};

// ─────────────────────────────────────────────
//  Monte Carlo result
// ─────────────────────────────────────────────
struct MCResult {
    double price;       // Discounted expected payoff
    double stdError;    // Standard error of the estimate
    double confLo;      // 95% confidence interval lower bound
    double confHi;      // 95% confidence interval upper bound
    Greeks greeks;      // Computed via finite-difference re-pricing
    long   pathsRun;    // Actual number of paths simulated
};

// ─────────────────────────────────────────────
//  Monte Carlo Engine
//
//  Supports European, Asian (arithmetic + geometric),
//  barrier (up/down, in/out), and lookback options.
//
//  Parallelisation: OpenMP parallel_for over independent path
//  blocks. Each thread maintains its own RNG state (seeded
//  deterministically from the base seed + thread id) so results
//  are reproducible regardless of thread count.
//
//  Variance reduction:
//    Antithetic variates   — pairs each path with its sign-flipped
//                            Brownian increments; halves variance for
//                            smooth payoffs at negligible cost.
//    Control variate       — uses the European BS price as a control;
//                            regresses out the known component of variance.
//    Both can be combined.
// ─────────────────────────────────────────────
class MonteCarloEngine {
public:
    // Full pricing with Greeks
    static MCResult price(const MCParams& p);

    // Price-only — faster, used internally for Greek bumping
    static MCResult priceOnly(const MCParams& p);

private:
    // Core path simulation — returns (sumPayoff, sumPayoff²) over a block of paths
    struct PathStats { double sum; double sumSq; long count; };

    static PathStats simulateBlock(
        const MCParams& p,
        int             blockPaths,
        uint64_t        threadSeed
    );

    // Compute payoff for a single completed path
    static double payoff(
        const MCParams&        p,
        const std::vector<double>& path   // log-prices along the path
    );

    // Xorshift64 — fast, good statistical properties, trivially seedable per-thread
    static double randNormal(uint64_t& state);
};

} // namespace pricing