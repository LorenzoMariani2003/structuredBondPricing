import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from discount_curve import calculate_curve, getContinousInterestRate
from montecarlo import simulate_paths_and_coupon

from FinDates.daycount import yearfrac #todo check implementation of this yearfrac, I assumed them to be correct _tom



BENCHMARK_OUTPUT = True

def main(Notional=1000000):        
    print("--------------------------------Question 1--------------------------------")
    print("Discount Curve Calculation and Structured Bond Pricing\n")
    discount_curve_schedule, D_today_settle, dtToday, dtStart = calculate_curve()
    print(f"D(Today->Start) = {D_today_settle:.10f}")
    
    D_0_4 = discount_curve_schedule.iloc[-1]
    last_date = discount_curve_schedule.index[-1]
    
    # Get continuous interest rate for T=4 years using the last discount factor
    #r_4y = -np.log(D_0_4) / 4.0 #todo fix with continous vector instead of fixed value
        
    seed=12


    print(f"\nLast date in discount curve: {last_date.date()}")
    #print(f"Continuous interest rate for T=4 years: {r_4y:.6%}")
    print(f"Discount factor for T=4 years: {D_0_4:.6f}")
    
    # All three components of the pricing formula:
    
    # Euribor component (1 - D_0_4)
    pv_euribor = 1.0 - D_0_4
    
    # Spread component (spread * dt * sum of discount factors)
    spread = 0.03
#    dt = 0.25 #todo compute actual dates


    dates = discount_curve_schedule.index
    dts = [yearfrac(dates[i-1],dates[i], "ACT/360") for i in range (1,len(dates))]

    dtSettle = pd.Timestamp("2023-02-02")
    

    dt_first = yearfrac(dtSettle, dates[0], "ACT/360")
    dts = [dt_first] + dts  

    pv_spread = spread * sum(df * dt for df, dt in zip(discount_curve_schedule.values, dts))
    

    D_vals = discount_curve_schedule.values  # considering 16 discount factors
    D_prev = np.concatenate([[D_today_settle], D_vals[:-1]]) 
    
    dts_years = np.array(dts)

    r_forward_quarterly = -np.log(D_vals / D_prev) / dts_years  



    
    # Coupon component (Monte Carlo simulation)

    pv_coupon, _ = simulate_paths_and_coupon(r_vec=r_forward_quarterly, D_0_4=D_0_4, N_sim=100000, seed=seed) 
    


    # Final equaation 
    Upfront_X = pv_euribor + pv_spread - pv_coupon
    
    print("\n--- PRICING ---")
    print(f"PV Euribor: {pv_euribor:.4f}")
    print(f"PV Spread:  {pv_spread:.4f}")
    print(f"PV Coupon:  {pv_coupon:.4f}")
    print(f"Upfront X%: {Upfront_X:.4f} ({Upfront_X * 100:.2f}%)")

    print("--------------------------------Question 2--------------------------------")
    pv_coupon_benel, _ = simulate_paths_and_coupon(r_vec=r_forward_quarterly, D_0_4=D_0_4,S0_enel=101, N_sim=100000,seed=seed) 
    
    # Final equaation for Bumped enel
    Upfront_X_benel = pv_euribor + pv_spread - pv_coupon_benel


    pv_coupon_baxa, _ = simulate_paths_and_coupon(r_vec=r_forward_quarterly, D_0_4=D_0_4,S0_axa=201, N_sim=100000, seed=seed) 
    
    # Final equaation for Bumped enel
    Upfront_X_baxa = pv_euribor + pv_spread - pv_coupon_baxa

    delta_enel = Upfront_X_benel - Upfront_X
    delta_axa = Upfront_X_baxa - Upfront_X

    print("\n--- Bump and revalue results ---")
    #to understand the below results: if delta_enel is positive, it means that each time enel gain 1 euros my devrivative gain(or lose) delta enel %
    print(f"Delta Upfront X% for Enel (S0=101): {delta_enel:.4f} ({delta_enel * 100:.2f}%)")
    print(f"Delta Upfront X% for Axa (S0=201): {delta_axa:.4f} ({delta_axa * 100:.2f}%)")


    T_values = np.array([yearfrac(dtSettle, d, "ACT/365 FIXED") for d in dates]) 
    

    bumped_discounts = np.array([
    np.exp(-(-np.log(D) / T + 0.0001) * T)
    for D, T in zip(D_vals, T_values)])
        
    D_0_4_bumped = bumped_discounts[-1]
    
    T_settle = yearfrac(
    dtToday,
    dtSettle,
    "ACT/365 FIXED"
    )

    D_today_settle_bumped = np.exp(
        -(-np.log(D_today_settle)/T_settle + 0.0001)
        * T_settle
    )

    D_prev_bumped = np.concatenate([[D_today_settle_bumped], bumped_discounts[:-1]])

    r_forward_quarterly_bumped = -np.log(bumped_discounts / D_prev_bumped) / dts_years


    # Euribor component (1 - D_0_4)
    pv_euribor_bumped = 1.0 - D_0_4_bumped
    
    # Spread component (spread * dt * sum of discount factors)
    spread = 0.03
    pv_spread_bumped  = spread * sum(df * dt for df, dt in zip(bumped_discounts, dts))

    
    # Coupon component (Monte Carlo simulation)

    pv_coupon_bumped, _ = simulate_paths_and_coupon(r_vec=r_forward_quarterly_bumped, D_0_4=D_0_4_bumped, N_sim=100000, seed=seed) 
    
    # Final equaation 
    Upfront_X_bumped = pv_euribor_bumped + pv_spread_bumped - pv_coupon_bumped
    dv01 = Upfront_X_bumped - Upfront_X

    print("\n--- PRICING ---")
    print(f"PV Euribor bumped: {pv_euribor_bumped:.4f}")
    print(f"PV Spread bumped:  {pv_spread_bumped:.4f}")
    print(f"PV Coupon bumped:  {pv_coupon_bumped:.4f}")
    print(f"Upfront bumped X%: {Upfront_X_bumped:.4f} ({Upfront_X_bumped * 100:.2f}%)")
    print(f"Delta Upfront X% for 1bp bump: {dv01:.6f} ({(Upfront_X_bumped - Upfront_X) * 100:.2f}%)")


    print("\n--------------------------------Question 3--------------------------------")
    print("--- Hedging ---")
    S0_enel = 100
    S0_axa = 200
    
    enel_NoS = -(delta_enel)*Notional
    axa_NoS = -(delta_axa)*Notional

    dv01_upfront = (Upfront_X_bumped - Upfront_X)

    dv01_swap_1eur = sum(df * dt for df, dt in zip(discount_curve_schedule.values, dts))  * 0.0001

    swap_notional = Notional*abs(dv01_upfront) / dv01_swap_1eur
    
    print(f"Hedge Delta ENEL: Buy {enel_NoS:.6f} shares per {Notional} EUR notional")
    print(f"Hedge Delta AXA:  Buy {axa_NoS:.6f} shares per {Notional} EUR notional")
    if dv01_upfront > 0:
            print(f"Hedge IR Risk:    Enter a Receiver IRS (pay 3m Euribor, receive fixed) with Notional = {swap_notional:.4f} EUR per {Notional} EUR notional")
    else:
        print(f"Hedge IR Risk:    Enter a Payer IRS (receive 3m Euribor, pay fixed) with Notional = {swap_notional:.4f} EUR per {Notional} EUR notional")

    if BENCHMARK_OUTPUT:
        print("\n--------------------------------Some checks--------------------------------")
        pv_coupon_closed_form, _ = simulate_paths_and_coupon(r_vec=r_forward_quarterly, D_0_4=D_0_4, N_sim=100000, seed=seed, benchmark=True)
if __name__ == "__main__":    main(Notional = 1000000)