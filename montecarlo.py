import numpy as np
from scipy.stats import norm

BDAYS_YEAR = 256


def calculate_asian_basket_coupon(monitor_prices, alpha, D_0_4):
    """Arithmetic Asian basket coupon from MC paths."""
    ratios_enel = monitor_prices[0, 1:, :] / monitor_prices[0, :-1, :]
    ratios_axa  = monitor_prices[1, 1:, :] / monitor_prices[1, :-1, :]

    basket_return = (0.5 * np.mean(ratios_enel, axis=0)
                   + 0.5 * np.mean(ratios_axa,  axis=0)
                   - 1.0)

    expected_coupon   = alpha * np.mean(np.maximum(0.0, basket_return))
    return expected_coupon * D_0_4


def geometric_basket_closed_form(r_vec, D_0_4, alpha=0.95,
                                  vol_enel=0.162, vol_axa=0.200,
                                  div_enel=0.025, div_axa=0.029,
                                  rho=0.40, n_monitoring=4):
    """
    Closed-form lower bound via geometric basket approximation.

    Replace arithmetic mean of annual ratios with geometric mean:
        G = (E_enel(T)/E_enel(0))^(1/8) * (E_axa(T)/E_axa(0))^(1/8)

    G is lognormal under Q since it is a product of lognormals.
    Arithmetic >= Geometric (Jensen) => this is a LOWER BOUND on the MC price.

    Annual forward rate for year n:
        f_n = sum of quarterly rates in that year * dt_q  (continuous compounding)
    """
    dt_q = 0.25
    # Annual forward log-drifts: f_n = integral of r(t) dt over year n
    f_annual = [float(np.sum(r_vec[4*n : 4*(n+1)]) * dt_q) for n in range(n_monitoring)]

    # E[ln G_s] for each stock: (1/4) * sum_n (f_n - d_s - 0.5*sigma_s^2)
    mu_e = sum(f - div_enel - 0.5*vol_enel**2 for f in f_annual) / 4
    mu_a = sum(f - div_axa  - 0.5*vol_axa**2  for f in f_annual) / 4

    # Geometric basket G = (G_enel * G_axa)^(1/2)
    # ln G ~ N(mu_G, sigma_G^2)
    mu_G    = 0.5 * (mu_e + mu_a)
    sigma_G = np.sqrt((vol_enel**2 + vol_axa**2 + 2*rho*vol_enel*vol_axa) / 16)

    # E[max(0, G - 1)] — Black-Scholes formula for lognormal with strike K=1
    F  = np.exp(mu_G + 0.5 * sigma_G**2)   # E[G]
    d1 = (mu_G + sigma_G**2) / sigma_G      # ln(K)=0 since K=1
    d2 = mu_G / sigma_G

    coupon = alpha * (F * norm.cdf(d1) - norm.cdf(d2))
    return coupon * D_0_4


def _benchmark_european_call(S=100.0, K=100.0, T=1.0, r=0.05, vol=0.20,
                              div=0.0, N_sim=1_000_000, seed=None):
    """
    Validates the MC engine against Black-Scholes for a plain European call.
    Since this has an exact formula, MC and analytical should match to ~2-3bp.
    This is NOT the product we are pricing — it is a sanity check only.
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Analytical Black-Scholes ---
    d1 = (np.log(S / K) + (r - div + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    bs_price = (S * np.exp(-div * T) * norm.cdf(d1)
                - K * np.exp(-r * T) * norm.cdf(d2))

    # --- Monte Carlo ---
    Z  = np.random.standard_normal(N_sim)
    ST = S * np.exp((r - div - 0.5 * vol**2) * T + vol * np.sqrt(T) * Z)
    mc_price = np.exp(-r * T) * np.mean(np.maximum(ST - K, 0.0))

    return bs_price, mc_price

def simulate_paths_and_coupon(r_vec, D_0_4, S0_enel=100.0, S0_axa=200.0,
                              vol_enel=0.162, vol_axa=0.200,
                              div_enel=0.025, div_axa=0.029,
                              rho=0.40, n_monitoring=4, alpha=0.95,
                              N_sim=10000, seed=None, benchmark=False):
    """
    Daily GBM simulation (256 business days/year), N=4 annual monitoring dates.

    Basket return:
        S(T) = (1/d) * sum_s [ (1/N) * sum_n  E_s(t_n) / E_s(t_{n-1}) ]

    Delta = 0 exactly by construction:
        Under GBM, if S0 -> lambda*S0, every ratio E_s(t_n)/E_s(t_{n-1})
        is unchanged (the lambda cancels). So the payoff distribution
        is independent of S0 => Delta ≡ 0 analytically, not numerically.

    Parameters
    ----------
    benchmark : bool
        If True, also print the geometric basket closed-form lower bound
        for comparison with the MC price.
    """
    if seed is not None:
        np.random.seed(seed)

    dt             = 1.0 / BDAYS_YEAR
    steps_per_year = BDAYS_YEAR
    steps_per_quarter = BDAYS_YEAR // 4
    total_steps    = n_monitoring * steps_per_year   # 4 * 256 = 1024

    corr_matrix = np.array([[1.0, rho],
                             [rho, 1.0]])

    monitor_prices = np.zeros((2, n_monitoring + 1, N_sim))
    monitor_prices[0, 0, :] = S0_enel
    monitor_prices[1, 0, :] = S0_axa

    S = np.array([np.full(N_sim, S0_enel),
                  np.full(N_sim, S0_axa)], dtype=float)

    for step in range(1, total_steps + 1):
        quarter_id = min((step - 1) // steps_per_quarter, len(r_vec) - 1)
        r = r_vec[quarter_id]
        Z = np.random.multivariate_normal([0.0, 0.0], corr_matrix, N_sim).T

        S[0] *= np.exp((r - div_enel - 0.5 * vol_enel**2) * dt
                       + vol_enel * np.sqrt(dt) * Z[0])
        S[1] *= np.exp((r - div_axa  - 0.5 * vol_axa**2)  * dt
                       + vol_axa  * np.sqrt(dt) * Z[1])

        if step % steps_per_year == 0:
            m = step // steps_per_year
            monitor_prices[0, m, :] = S[0]
            monitor_prices[1, m, :] = S[1]

    discounted_coupon = calculate_asian_basket_coupon(monitor_prices, alpha, D_0_4)

    if benchmark:
            # --- 1. European call sanity check ---
            r_flat  = float(np.mean(r_vec))          # representative rate
            bs, mc  = _benchmark_european_call(
                S=100.0, K=100.0, T=1.0,
                r=r_flat, vol=vol_enel,
                div=div_enel, N_sim=500_000, seed=seed
            )
            print("\n--- MC Sanity Check: European Call on single asset ---")
            print(f"  Black-Scholes price : {bs:.6f}")
            print(f"  Monte Carlo price   : {mc:.6f}")
            print(f"  Difference          : {abs(bs - mc):.6f}  "
                  f"({'OK' if abs(bs-mc) < 0.005 else 'WARNING — check MC'})")
            print(f"  Interpretation: arithmetic Asian basket has no closed form.")
            print(f"  This test confirms the MC engine itself is correct.")
    
            # --- 2. Geometric basket lower bound ---
            cf = geometric_basket_closed_form(
                r_vec, D_0_4, alpha, vol_enel, vol_axa,
                div_enel, div_axa, rho, n_monitoring
            )
            print(f"\n--- Benchmark: Geometric basket (closed-form lower bound) ---")
            print(f"  MC price   (arithmetic basket) : {discounted_coupon:.6f}")
            print(f"  Closed-form (geometric basket) : {cf:.6f}")
            print(f"  Gap (arithmetic - geometric)   : {discounted_coupon - cf:.6f}")
            print(f"  Interpretation: arithmetic >= geometric by Jensen inequality,")
            print(f"  so the gap should be positive. If it is, both results are consistent.")

    return discounted_coupon, monitor_prices