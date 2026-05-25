import numpy as np

BDAYS_YEAR = 256  # business days per year (consistent with Utilities.py)


def calculate_asian_basket_coupon(monitor_prices, alpha, D_0_4):
    # Basket return = equally-weighted mean of ratios minus 1
    ratios_enel = monitor_prices[0, 1:, :] / monitor_prices[0, :-1, :]
    ratios_axa  = monitor_prices[1, 1:, :] / monitor_prices[1, :-1, :]

    basket_return = (0.5 * np.mean(ratios_enel, axis=0)
                   + 0.5 * np.mean(ratios_axa,  axis=0)
                   - 1.0)

    expected_coupon   = alpha * np.mean(np.maximum(0.0, basket_return))
    discounted_coupon = expected_coupon * D_0_4

    return discounted_coupon

def simulate_paths_and_coupon(r_vec, D_0_4, S0_enel=100.0, S0_axa=200.0,
                              vol_enel=0.162, vol_axa=0.200,
                              div_enel=0.025, div_axa=0.029,
                              rho=0.40, n_monitoring=4, alpha=0.95, N_sim=10000, seed=None):
    """
    Daily GBM simulation (256 business days/year) with N=4 annual monitoring dates.

    Basket return follows the termsheet definition:
        S(T) = (1/d) * sum_s [ (1/N) * sum_n  E_s(t_n) / E_s(t_{n-1}) ]
    where the N=4 monitoring dates are t1=1y, t2=2y, t3=3y, t4=4y=T. 

    Since each ratio E_s(t_n)/E_s(t_{n-1}) is independent of S0,
    the equity delta of the coupon is exactly zero by construction 
    
    #tom ? non mi sembra giusto

    Returns
    -------
    discounted_coupon : float
        B(0,T) * alpha * E^Q[ max(0, basket_return) ]
    monitor_prices : np.ndarray, shape (2, n_monitoring+1, N_sim)
        Prices at t0 and at each of the 4 annual monitoring dates.
    """
    if seed is not None:
        np.random.seed(seed)


    dt = 1.0 / BDAYS_YEAR          # one business day
    steps_per_year = BDAYS_YEAR
    steps_per_quarter= BDAYS_YEAR // 4
    total_steps = n_monitoring * steps_per_year   # 4 * 256 = 1024

    corr_matrix = np.array([[1.0, rho],
                             [rho, 1.0]])

    # Store only t0 + 4 monitoring dates (memory efficient — no daily storage)
    monitor_prices = np.zeros((2, n_monitoring + 1, N_sim))
    monitor_prices[0, 0, :] = S0_enel
    monitor_prices[1, 0, :] = S0_axa

    # Propagate daily, snapshot at annual checkpoints
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
            m = step // steps_per_year          # 1, 2, 3, 4
            monitor_prices[0, m, :] = S[0]
            monitor_prices[1, m, :] = S[1]

    discounted_coupon = calculate_asian_basket_coupon(monitor_prices, alpha, D_0_4)

    return discounted_coupon, monitor_prices
import matplotlib.pyplot as plt
r_test = np.full(16, 0.02)
prices = simulate_paths_and_coupon(r_vec=r_test, D_0_4=0.95, N_sim=100)[1]
prices_enel = prices[0, :, :]
prices_axa = prices[1, :, :]

plt.figure(figsize=(10, 5))
plt.plot(prices_enel, linestyle="-")
plt.title("Brownian Motion of Enel Stock Price")
plt.xlabel("Trimester")
plt.ylabel("Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("enel.png")


plt.figure(figsize=(10, 5))
plt.plot(prices_axa, linestyle="-")
plt.title("Brownian Motion of Axa Stock Price")
plt.xlabel("Trimester")
plt.ylabel("Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("axa.png")