# Structured Bond Pricing

This project prices and risk-manages a 4-year capital-protected structured note linked to an Enel and Axa equity basket (Asian). It builds a risk-free discount curve via bootstrapping, simulates correlated stock paths using Monte Carlo Geometric Brownian Motion to price the coupon, computes Equity Delta and Interest Rate DV01 via bump-and-revalue, and determines the shares and IRS notional required for a complete hedging strategy.

## Important features to check before running
If you want to run the algorithm with the official asian basket (t_{n-1} in the denominator) make sure that montecarlo.py in calculate_asian_basket_coupon, use_t0_denominator is set as false.
If you don't want to visualize benchmark checks just set to false BENCHMARK_OUTPUT in main.py


## Installation and Usage

Clone the repository, install the required dependencies, and execute the main model:

```bash
git clone [https://github.com/LorenzoMariani2003/structuredBondPricing.git](https://github.com/LorenzoMariani2003/structuredBondPricing.git)
cd structuredBondPricing
pip install -r requirements.txt
python main.py
```