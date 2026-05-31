import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm




# ---------------------------------------------------------------------------
# Payoff helpers
# ---------------------------------------------------------------------------

def calculate_asian_basket_coupon(monitor_prices, alpha, D_0_4, use_t0_denominator=False):
    """
    Arithmetic Asian basket coupon from MC paths.

    Basket return:
        standard (use_t0_denominator=False):
            B = 0.5 * mean_n(E_enel(t_n)/E_enel(t_{n-1}))
              + 0.5 * mean_n(E_axa(t_n) /E_axa(t_{n-1}))
              - 1
        use_t0_denominator=True:
            B = 0.5 * mean_n(E_enel(t_n)/E_enel(t_0))
              + 0.5 * mean_n(E_axa(t_n) /E_axa(t_0))
              - 1
    Coupon PV = alpha * E[max(0, B)] * D(0, T)
    """
    if use_t0_denominator:
        S0_enel = monitor_prices[0, 0:1, :]   # shape (1, N_sim) — broadcasts over 4 dates
        S0_axa  = monitor_prices[1, 0:1, :]
        ratios_enel = monitor_prices[0, 1:, :] / S0_enel
        ratios_axa  = monitor_prices[1, 1:, :] / S0_axa
    else:
        ratios_enel = monitor_prices[0, 1:, :] / monitor_prices[0, :-1, :]
        ratios_axa  = monitor_prices[1, 1:, :] / monitor_prices[1, :-1, :]

    basket_return = (0.5 * np.mean(ratios_enel, axis=0)
                   + 0.5 * np.mean(ratios_axa,  axis=0)
                   - 1.0)

    expected_coupon = alpha * np.mean(np.maximum(0.0, basket_return))
    return expected_coupon * D_0_4


def calculate_geometric_basket_coupon_mc(monitor_prices, alpha, D_0_4):
    """
    Geometric basket coupon priced from the same MC paths.

    Replace the arithmetic mean of annual ratios with the geometric mean:
        G_s = (prod_n E_s(t_n)/E_s(t_{n-1}))^(1/N) = (E_s(T)/E_s(0))^(1/N)
        G   = sqrt(G_enel * G_axa)   (equal-weight geometric basket)

    Coupon PV = alpha * E[max(0, G - 1)] * D(0, T)

    Because G is lognormal under Q, this MC estimate can be directly compared
    against the closed-form price in geometric_basket_closed_form -- providing
    a self-consistent validation of the simulation engine using the very same
    paths generated for the actual product.
    """
    n = monitor_prices.shape[1] - 1          # number of monitoring dates (4)

    geom_enel = (monitor_prices[0, -1, :] / monitor_prices[0, 0, :]) ** (1.0 / n)
    geom_axa  = (monitor_prices[1, -1, :] / monitor_prices[1, 0, :]) ** (1.0 / n)

    basket_return = np.sqrt(geom_enel * geom_axa) - 1.0

    expected_coupon = alpha * np.mean(np.maximum(0.0, basket_return))
    return expected_coupon * D_0_4


# ---------------------------------------------------------------------------
# Closed-form reference
# ---------------------------------------------------------------------------

def geometric_basket_closed_form(r_vec, D_0_4, alpha=0.95,
                                  vol_enel=0.162, vol_axa=0.200,
                                  div_enel=0.025, div_axa=0.029,
                                  rho=0.40, n_monitoring=4):
    """
    Closed-form price for the geometric basket coupon.

    Under Q, each annual ratio E_s(t_n)/E_s(t_{n-1}) is lognormal with:
        log-drift : f_n - d_s - 0.5*sigma_s^2
        log-vol   : sigma_s * sqrt(1 year)

    The geometric mean G_s = (prod_n ratio_n)^(1/N) is also lognormal:
        ln G_s ~ N(mu_s, sigma_s^2/N)  where mu_s = (1/N)*sum_n(f_n - d_s - 0.5*sigma_s^2)

    The geometric basket G = sqrt(G_enel * G_axa) is lognormal:
        mu_G    = 0.5*(mu_enel + mu_axa)
        sigma_G = sqrt((vol_enel^2 + vol_axa^2 + 2*rho*vol_enel*vol_axa) / (4*N))

    E[max(0, G-1)] is priced by the Black-Scholes formula with K=1, F=E[G].

    Jensen inequality => arithmetic basket >= geometric basket, so this price
    is a LOWER BOUND on the arithmetic Asian basket MC price.
    """
    dt_q = 0.25
    f_annual = [float(np.sum(r_vec[4*n : 4*(n+1)]) * dt_q) for n in range(n_monitoring)]

    mu_e = sum(f - div_enel - 0.5 * vol_enel**2 for f in f_annual) / n_monitoring
    mu_a = sum(f - div_axa  - 0.5 * vol_axa**2  for f in f_annual) / n_monitoring

    mu_G    = 0.5 * (mu_e + mu_a)
    sigma_G = np.sqrt((vol_enel**2 + vol_axa**2 + 2 * rho * vol_enel * vol_axa)
                      / (4 * n_monitoring))

    F  = np.exp(mu_G + 0.5 * sigma_G**2)
    d1 = (mu_G + sigma_G**2) / sigma_G
    d2 = mu_G / sigma_G

    coupon = alpha * (F * norm.cdf(d1) - norm.cdf(d2))
    return coupon * D_0_4


# ---------------------------------------------------------------------------
# Fast terminal sampler  (convergence analysis only)
# ---------------------------------------------------------------------------

def _simulate_geometric_basket_fast(r_vec, D_0_4, alpha,
                                     vol_enel, vol_axa, div_enel, div_axa,
                                     rho, n_monitoring, N_sim, seed=None):
    """
    Fast geometric basket pricer via direct terminal sampling.

    G_s = (E_s(T)/E_s(0))^(1/N) is lognormal under Q, so we sample
    ln(E_s(T)/E_s(0)) directly without daily time-stepping -- valid because
    the geometric basket payoff depends only on terminal prices, not the path.

    Returns
    -------
    price : float
        Discounted geometric basket coupon estimate.
    stderr : float
        MC standard error (1-sigma) of the estimate.
    """
    if seed is not None:
        np.random.seed(seed)

    dt_q = 0.25
    f_annual = [float(np.sum(r_vec[4*n : 4*(n+1)]) * dt_q) for n in range(n_monitoring)]

    # Total log-drift and log-vol over the full 4-year horizon
    mu_e_4  = sum(f - div_enel - 0.5 * vol_enel**2 for f in f_annual)
    mu_a_4  = sum(f - div_axa  - 0.5 * vol_axa**2  for f in f_annual)
    sig_e_4 = vol_enel * np.sqrt(float(n_monitoring))
    sig_a_4 = vol_axa  * np.sqrt(float(n_monitoring))

    cov = np.array([[sig_e_4**2,              rho * sig_e_4 * sig_a_4],
                    [rho * sig_e_4 * sig_a_4, sig_a_4**2             ]])
    Z = np.random.multivariate_normal([0.0, 0.0], cov, N_sim).T

    # ln(E_s(T)/E_s(0)) for each simulated path
    ln_ratio_enel = mu_e_4 + Z[0]
    ln_ratio_axa  = mu_a_4 + Z[1]

    # G_s = ratio^(1/N)  =>  ln G_s = ln_ratio / N
    geom_enel = np.exp(ln_ratio_enel / n_monitoring)
    geom_axa  = np.exp(ln_ratio_axa  / n_monitoring)

    payoffs = alpha * np.maximum(0.0, np.sqrt(geom_enel * geom_axa) - 1.0)

    price  = np.mean(payoffs) * D_0_4
    stderr = np.std(payoffs, ddof=1) / np.sqrt(N_sim) * D_0_4
    return price, stderr


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_geometric_basket_convergence(r_vec, D_0_4, alpha=0.95,
                                       vol_enel=0.162, vol_axa=0.200,
                                       div_enel=0.025, div_axa=0.029,
                                       rho=0.40, n_monitoring=4,
                                       N_values=None, n_reps=20,
                                       base_seed=0,
                                       save_path="geom_basket_convergence.png"):
    """
    Convergence study for the geometric basket MC pricer.

    For each N in N_values, n_reps independent replications (different seeds)
    are run and the sample mean plus a +/-2-sigma band across replications is
    plotted against the closed-form reference line.

    Because the geometric basket has an exact closed form, convergence of the
    MC estimates toward that reference directly validates the simulation engine.

    Parameters
    ----------
    N_values : list[int], optional
        Simulation sizes to sweep. Defaults to a log-spaced grid 500 -> 100k.
    n_reps : int
        Independent replications per N level (drives error-bar width).
    base_seed : int
        Replication i uses seed = base_seed + i.
    save_path : str
        Destination path for the saved PNG figure.

    Returns
    -------
    means, lows, highs : np.ndarray
        Per-N mean and +/-2-sigma bounds across replications.
    cf_price : float
        Closed-form reference price.
    """
    if N_values is None:
        N_values = [500, 1_000, 2_000, 5_000, 10_000, 30_000, 100_000]

    cf_price = geometric_basket_closed_form(
        r_vec, D_0_4, alpha, vol_enel, vol_axa,
        div_enel, div_axa, rho, n_monitoring
    )

    means, lows, highs = [], [], []

    for N in N_values:
        rep_prices = np.array([
            _simulate_geometric_basket_fast(
                r_vec, D_0_4, alpha, vol_enel, vol_axa,
                div_enel, div_axa, rho, n_monitoring,
                N_sim=N, seed=base_seed + i
            )[0]
            for i in range(n_reps)
        ])
        m = np.mean(rep_prices)
        s = np.std(rep_prices, ddof=1)
        means.append(m)
        lows.append(m - 2 * s)
        highs.append(m + 2 * s)

    means = np.array(means)
    lows  = np.array(lows)
    highs = np.array(highs)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.fill_between(N_values, lows, highs, alpha=0.20, color="steelblue",
                    label=r"MC mean $\pm 2\sigma$ (across replications)")
    ax.errorbar(N_values, means,
                yerr=[means - lows, highs - means],
                fmt="o-", color="steelblue", capsize=4, linewidth=1.5,
                label="MC mean (geometric basket)")
    ax.axhline(cf_price, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Closed-form reference ({cf_price:.6f})")

    ax.set_xscale("log")
    ax.set_xlabel("N simulations")
    ax.set_ylabel("Coupon PV")
    ax.set_title("MC Convergence: geometric basket coupon vs closed-form")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Convergence plot saved to '{save_path}'")

    return means, lows, highs, cf_price

# In montecarlo.py o in discount_curve.py

from Utilities import Calendar

def count_business_days(year: int, calendar_code: str = "de.eurex") -> int:
    """Used to count business days in a given year according to the specified calendar."""
    import datetime
    cal = Calendar(calendar_code)
    count = 0
    day = datetime.date(year, 1, 1)
    while day.year == year:
        if not cal.is_holiday(day):
            count += 1
        day += datetime.timedelta(days=1)
    return count

def avg_business_days(start_year: int, n_years: int = 4, 
                      calendar_code: str = "de.eurex") -> float:
    """Avg business days per year over a range of years, used for discounting in the MC simulation."""
    counts = [count_business_days(start_year + i, calendar_code) 
              for i in range(n_years)]
    return sum(counts) / len(counts)
# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------

BDAYS_YEAR = round(avg_business_days(2023, n_years=4))
print(f"Average business days per year (2023-2026, de.eurex calendar): {BDAYS_YEAR}")

def simulate_paths_and_coupon(r_vec, D_0_4, S0_enel=100.0, S0_axa=200.0,
                              vol_enel=0.162, vol_axa=0.200,
                              div_enel=0.025, div_axa=0.029,
                              rho=0.40, n_monitoring=4, alpha=0.95,
                              N_sim=10_000, seed=None, benchmark=False):
    """
    Daily GBM simulation (256 business days/year), N=4 annual monitoring dates.

    Basket return (arithmetic Asian):
        B = (1/d) * sum_s [ (1/N) * sum_n  E_s(t_n) / E_s(t_{n-1}) ]  - 1

    Delta = 0 exactly by construction:
        Under GBM, if S0 -> lambda*S0, every ratio E_s(t_n)/E_s(t_{n-1})
        is unchanged (the lambda cancels). So the payoff distribution
        is independent of S0 => Delta == 0 analytically, not numerically.

    Parameters
    ----------
    r_vec : array-like, length 16
        Quarterly forward rates (continuously compounded).
    D_0_4 : float
        Discount factor from today to maturity T=4y.
    benchmark : bool
        If True:
          1. Prices the geometric basket on the SAME daily paths and compares
             with the closed-form solution (point-estimate consistency check).
          2. Runs a convergence study via fast terminal sampling across a range
             of N values and saves a plot with +/-2-sigma confidence bands.
        Both tests validate the MC engine without any auxiliary simulation.
    """
    if seed is not None:
        np.random.seed(seed)

    dt                = 1.0 / BDAYS_YEAR
    steps_per_year    = BDAYS_YEAR
    steps_per_quarter = BDAYS_YEAR // 4
    total_steps       = n_monitoring * steps_per_year   # 4 * 256 = 1024

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
        # ------------------------------------------------------------------
        # 1. Point-estimate check: geometric basket on same paths vs closed-form
        # ------------------------------------------------------------------
        mc_geom = calculate_geometric_basket_coupon_mc(monitor_prices, alpha, D_0_4)
        cf_geom = geometric_basket_closed_form(
            r_vec, D_0_4, alpha, vol_enel, vol_axa,
            div_enel, div_axa, rho, n_monitoring
        )

        print("\n--- Benchmark: geometric basket (MC on same paths vs closed-form) ---")
        print(f"  MC price    (geometric basket) : {mc_geom:.6f}")
        print(f"  Closed-form (geometric basket) : {cf_geom:.6f}")
        print(f"  Difference  (MC - closed-form) : {mc_geom - cf_geom:.6f}")
        print(f"  Interpretation: close agreement validates the MC engine.")
        print(f"  The arithmetic basket (product) price is above by Jensen inequality:")
        print(f"  MC price    (arithmetic basket): {discounted_coupon:.6f}")
        print(f"  Gap (arithmetic - geometric)   : {discounted_coupon - mc_geom:.6f}"
              f"  [should be > 0]")

        # ------------------------------------------------------------------
        # 2. Convergence study across N sizes (fast terminal sampler)
        # ------------------------------------------------------------------
        print("\n--- Convergence study: geometric basket MC vs closed-form ---")
        print("  Running 20 replications per N level "
              "across [500, 1k, 2k, 5k, 10k, 30k, 100k] ...")

        plot_geometric_basket_convergence(
            r_vec=r_vec, D_0_4=D_0_4, alpha=alpha,
            vol_enel=vol_enel, vol_axa=vol_axa,
            div_enel=div_enel, div_axa=div_axa,
            rho=rho, n_monitoring=n_monitoring,
            n_reps=20, base_seed=0,
            save_path="geom_basket_convergence.png"
        )

    return discounted_coupon, monitor_prices