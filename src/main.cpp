#include "black_scholes.h"
#include <iostream>
#include <iomanip>

using namespace pricing;

void printResult(const std::string& label, const BSParams& p) {
    const auto result = BlackScholesEngine::price(p);
    const auto& g = result.greeks;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n── " << label << " ──\n";
    std::cout << "  Price  : " << result.price     << "\n";
    std::cout << "  Delta  : " << g.delta           << "\n";
    std::cout << "  Gamma  : " << g.gamma           << "\n";
    std::cout << "  Theta  : " << g.theta << " /day\n";
    std::cout << "  Vega   : " << g.vega  << " /1%vol\n";
    std::cout << "  Rho    : " << g.rho   << " /1%rate\n";
}

int main() {
    // ATM European Call
    BSParams call{100.0, 100.0, 0.05, 0.02, 0.20, 1.0, OptionType::Call};
    printResult("ATM European Call  S=100 K=100 r=5% q=2% σ=20% T=1Y", call);

    // OTM European Put
    BSParams put{100.0, 90.0, 0.05, 0.0, 0.25, 0.5, OptionType::Put};
    printResult("OTM European Put   S=100 K=90  r=5% q=0% σ=25% T=6M", put);

    // Implied volatility round-trip
    const double mkt = BlackScholesEngine::price(call).price;
    const double iv  = BlackScholesEngine::impliedVol(
        mkt, call.S, call.K, call.r, call.q, call.T, OptionType::Call);
    std::cout << "\n── IV Round-Trip ──\n";
    std::cout << "  Market price : " << mkt << "\n";
    std::cout << "  Solved IV    : " << iv  << "  (input σ = " << call.sigma << ")\n";

    return 0;
}