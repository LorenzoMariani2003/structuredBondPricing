import numpy as np

def simulate_paths_and_coupon(r, D_0_4, S0_enel=100.0, S0_axa=200.0, 
                              vol_enel=0.162, vol_axa=0.200, 
                              div_enel=0.025, div_axa=0.029, 
                              rho=0.40, dt=0.25, n_steps=16,strike_enel=100.0, strike_axa=200.0, alpha = 0.95, N_sim=10000):
    
    # Correlation matrix for the two stocks
    corr_matrix = np.array([[1.0, rho], 
                            [rho, 1.0]])
    
    # Array to store prices at each quarter: shape (2 assets, 17 time steps, N_sim)
    # Index 0 is time t=0, indices 1 to 16 are the quarterly steps
    prices = np.zeros((2, n_steps + 1, N_sim))
    prices[0, 0, :] = S0_enel
    prices[1, 0, :] = S0_axa
    
    # Random Walk step-by-step for each quarter
    for t in range(1, n_steps + 1):
        Z = np.random.multivariate_normal([0, 0], corr_matrix, N_sim).T
        
        prices[0, t, :] = prices[0, t-1, :] * np.exp((r - div_enel - 0.5 * vol_enel**2) * dt + vol_enel * np.sqrt(dt) * Z[0])
        prices[1, t, :] = prices[1, t-1, :] * np.exp((r - div_axa - 0.5 * vol_axa**2) * dt + vol_axa * np.sqrt(dt) * Z[1])
        
    # Final Payoff calculation at maturity (t=16)
    
    avg_price_enel = np.mean(prices[0, 1:, :], axis=0)
    avg_price_axa = np.mean(prices[1, 1:, :], axis=0)
    
    basket_return = 0.5 * (avg_price_enel / strike_enel - 1) + 0.5 * (avg_price_axa / strike_axa - 1)

    expected_coupon = alpha * np.mean(np.maximum(0, basket_return))
    
    discounted_coupon = expected_coupon * D_0_4
    
    return discounted_coupon, prices

#import matplotlib.pyplot as plt
#prices = simulate_paths_and_coupon(r=0.02, D_0_4=0.95)[1]
#prices_enel = prices[0, :, :]
#prices_axa = prices[1, :, :]
#
#plt.figure(figsize=(10, 5))
#plt.plot(prices_enel, linestyle="-")
#plt.title("Brownian Motion of Enel Stock Price")
#plt.xlabel("Trimester")
#plt.ylabel("Price")
#plt.grid(True)
#plt.tight_layout()
#plt.savefig("enel.png")
#
#
#plt.figure(figsize=(10, 5))
#plt.plot(prices_axa, linestyle="-")
#plt.title("Brownian Motion of Axa Stock Price")
#plt.xlabel("Trimester")
#plt.ylabel("Price")
#plt.grid(True)
#plt.tight_layout()
#plt.savefig("axa.png")