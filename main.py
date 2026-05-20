import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from discount_curve import calculate_curve, getContinousInterestRate
from montecarlo import simulate_paths_and_coupon


def main(N=1000000):        
    print("--------------------------------Question 1--------------------------------")
    print("Discount Curve Calculation and Structured Bond Pricing\n")
    discount_curve_schedule = calculate_curve()
    
    D_0_4 = discount_curve_schedule.iloc[-1]
    last_date = discount_curve_schedule.index[-1]
    
    # Get continuous interest rate for T=4 years using the last discount factor
    r_4y = -np.log(D_0_4) / 4.0
    
    print(f"\nLast date in discount curve: {last_date.date()}")
    print(f"Continuous interest rate for T=4 years: {r_4y:.6%}")
    print(f"Discount factor for T=4 years: {D_0_4:.6f}")
    
    # All three components of the pricing formula:
    
    # Euribor component (1 - D_0_4)
    pv_euribor = 1.0 - D_0_4
    
    # Spread component (spread * dt * sum of discount factors)
    spread = 0.03
    dt = 0.25
    sum_dfs = discount_curve_schedule.sum()
    pv_spread = sum_dfs * spread * dt
    
    # Coupon component (Monte Carlo simulation)

    pv_coupon, _ = simulate_paths_and_coupon(r=r_4y, D_0_4=D_0_4, N_sim=100000) 
    
    # Final equaation 
    Upfront_X = pv_euribor + pv_spread - pv_coupon
    
    print("\n--- PRICING ---")
    print(f"PV Euribor: {pv_euribor:.4f}")
    print(f"PV Spread:  {pv_spread:.4f}")
    print(f"PV Coupon:  {pv_coupon:.4f}")
    print(f"Upfront X%: {Upfront_X:.4f} ({Upfront_X * 100:.2f}%)")

    print("--------------------------------Question 2--------------------------------")
    pv_coupon_benel, _ = simulate_paths_and_coupon(r=r_4y, D_0_4=D_0_4,S0_enel=101, N_sim=100000) 
    
    # Final equaation for Bumped enel
    Upfront_X_benel = pv_euribor + pv_spread - pv_coupon_benel


    pv_coupon_baxa, _ = simulate_paths_and_coupon(r=r_4y, D_0_4=D_0_4,S0_axa=201, N_sim=100000) 
    
    # Final equaation for Bumped enel
    Upfront_X_baxa = pv_euribor + pv_spread - pv_coupon_baxa

    delta_enel = Upfront_X_benel - Upfront_X
    delta_axa = Upfront_X_baxa - Upfront_X

    print("\n--- Bump and revalue results ---")
    #to understand the below results: if delta_enel is positive, it means that each time enel gain 1 euros my devrivative gain(or lose) delta enel %
    print(f"Delta Upfront X% for Enel (S0=101): {delta_enel:.4f} ({delta_enel * 100:.2f}%)")
    print(f"Delta Upfront X% for Axa (S0=201): {delta_axa:.4f} ({delta_axa * 100:.2f}%)")


    bumped_discounts = []
    T_values = np.arange(0.25, 4.25, 0.25)
    for discount, T in zip(discount_curve_schedule.values, T_values):
        r_bumped = -np.log(discount) / T + 0.0001  
        D_bumped = np.exp(-r_bumped * T)  
        bumped_discounts.append(D_bumped)
        
    D_0_4_bumped = bumped_discounts[-1]  # Use the last bumped discount factor for T=4 years
    r_4y_bumped = -np.log(D_0_4_bumped) / 4.0

    # Euribor component (1 - D_0_4)
    pv_euribor_bumped = 1.0 - D_0_4_bumped
    
    # Spread component (spread * dt * sum of discount factors)
    spread = 0.03
    dt = 0.25
    sum_dfs_bumped = sum(bumped_discounts)
    pv_spread_bumped = sum_dfs_bumped * spread * dt
    
    # Coupon component (Monte Carlo simulation)

    pv_coupon_bumped, _ = simulate_paths_and_coupon(r=r_4y_bumped, D_0_4=D_0_4_bumped, N_sim=100000) 
    
    # Final equaation 
    Upfront_X_bumped = pv_euribor_bumped + pv_spread_bumped - pv_coupon_bumped
    
    print("\n--- PRICING ---")
    print(f"PV Euribor bumped: {pv_euribor_bumped:.4f}")
    print(f"PV Spread bumped:  {pv_spread_bumped:.4f}")
    print(f"PV Coupon bumped:  {pv_coupon_bumped:.4f}")
    print(f"Upfront bumped X%: {Upfront_X_bumped:.4f} ({Upfront_X_bumped * 100:.2f}%)")
    print(f"Delta Upfront X% for 1bp bump: {Upfront_X_bumped - Upfront_X:.6f} ({(Upfront_X_bumped - Upfront_X) * 100:.2f}%)")


    print("\n--------------------------------Question 3--------------------------------")
    print("--- Hedging ---")
    
    enel_NoF = abs(delta_enel)*N
    axa_NoF = abs(delta_axa)*N
    dv01_upfront = (Upfront_X_bumped - Upfront_X)

    dv01_swap_1eur = sum_dfs * 0.0001 * dt
    swap_notional = N*abs(dv01_upfront) / dv01_swap_1eur
    
    print(f"Hedge Delta ENEL: Buy {enel_NoF:.6f} shares per {N} EUR notional")
    print(f"Hedge Delta AXA:  Buy {axa_NoF:.6f} shares per {N} EUR notional")
    print(f"Hedge IR Risk:    Enter a Payer IRS (receive 3m Euribor, pay fixed) with Notional = {swap_notional:.4f} EUR per {N} EUR notional")

if __name__ == "__main__":    main(N = 1000000)