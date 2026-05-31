#Various imports from utilities and bootstrap files, as well as standard libraries for data manipulation and plotting.
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bootstrap import (
    DC_CONV,
    depo_converter,
    future_converter,
    swap_converter,
    bootstrapDepo,
    bootstrapFuture,
    bootstrapSwap,
    getZeroRates,
    getRatesLinInterpDiscount,
)
from Utilities import createSchedule
from FinDates.daycount import yearfrac


def import_data(data_dir):
    dt = pd.read_csv(
        data_dir / "dt.csv",
        index_col="Market",
        usecols=["Market", "TARGET"],
        converters={"TARGET": pd.to_datetime},
    )

    df_depo = pd.read_csv(
        data_dir / "depos.csv",
        index_col="Depos",
        usecols=["Depos", "ASK", "BID"],
        converters={"Depos": pd.to_datetime, "BID": depo_converter, "ASK": depo_converter},
    )

    futures = pd.read_csv(
        data_dir / "futures.csv",
        index_col="Future",
        usecols=["Future", "ASK", "BID"],
        converters={"Future": pd.to_datetime, "BID": future_converter, "ASK": future_converter},
    )

    settle = pd.read_csv(
        data_dir / "settles.csv",
        index_col="Future",
        usecols=["Future", "Settle", "Expiry"],
        converters={"Future": pd.to_datetime, "Settle": pd.to_datetime, "Expiry": pd.to_datetime},
    )

    df_futures = futures.join(settle)

    df_swaps = pd.read_csv(
        data_dir / "swaps.csv",
        index_col="Swap",
        usecols=["Swap", "BID", "ASK"],
        converters={"Swap": pd.to_datetime, "BID": swap_converter, "ASK": swap_converter},
    )

    return dt, df_depo, df_futures, df_swaps, settle


# Function to calculate the discount curve using the bootstrapping methods defined in the imported bootstrap file.  
def calculate_curve():
    

    base_path = Path.cwd()
    sys.path.insert(0, str(base_path))
    pricing_timestamp = pd.Timestamp("2023-01-31 10:45:00", tz="Europe/Rome")
    print("Pricing timestamp:", pricing_timestamp)
    data_dir = base_path / "data"

    
    dt, df_depo, df_futures, df_swaps, settle = import_data(data_dir)

    dtSettle = dt.loc["Settlement", "TARGET"]
    dtToday = dt.loc["Today", "TARGET"]
    



    try:
        sn_maturity = dtSettle + pd.Timedelta(days=1)  # 2023-02-03
        if sn_maturity not in df_depo.index:
            raise KeyError(f"SN maturity {sn_maturity.date()} not found in df_depo.index")

        r_sn_mid = 0.5 * (float(df_depo.loc[sn_maturity, "BID"]) + float(df_depo.loc[sn_maturity, "ASK"]))
        tau_gap = yearfrac(dtToday, dtSettle, "ACT/360")
        D_today_settle = 1.0 / (1.0 + r_sn_mid * tau_gap)
    except Exception as e:
        print("WARNING: could not compute D_today_settle from shortest deposit. "
          "Falling back to 1.0. Error:", repr(e)) #should never fall into this exception but I0ll leve it just in case
        D_today_settle = 1.0 


    # `df_futures` is already created in `import_data`; use first 7 points if needed
    if len(df_futures) > 7:
        df_futures = df_futures.iloc[:7]


    termDates = [dtSettle]
    discounts = [1.0]


    termDates, discounts = bootstrapDepo(dtSettle, df_depo, df_futures, termDates, discounts)
    termDates, discounts = bootstrapFuture(dtSettle, df_futures, termDates, discounts)
    termDates, discounts = bootstrapSwap(dtSettle, df_swaps, termDates, discounts)

    # Exact quarterly payment dates using Utilities.createSchedule from dtSettle = 2 Feb 2023
    periods = [f"{3*k}M" for k in range(1, 17)]
    schedule_dates = createSchedule(dtSettle, periods, calendarCode="de.eurex", adjustmentRule="follow")
    payment_dates = [pd.Timestamp(d) for d in schedule_dates]

    discounts_on_schedule = [
        getRatesLinInterpDiscount(dtSettle, pd.Timestamp(d), termDates, discounts)
        for d in payment_dates
    ]

        # Convert P(Settle, T) -> P(Today, T) = P(Today, Settle) * P(Settle, T)
    discounts_on_schedule = [D_today_settle * D for D in discounts_on_schedule]

    discount_curve_schedule = pd.Series(index=payment_dates, data=discounts_on_schedule)

    return discount_curve_schedule, D_today_settle, dtToday, dtSettle


def getContinousInterestRate(D,T):
    """Calculate the continuous interest rate for a given discount factor D and time T."""
    return -np.log(D) / T if T > 0 and D > 0 else 0.0



#print("Calculating discount curve...")
#discount_curve_schedule = calculate_curve()
#print("Discount curve calculated successfully.")
#print("Interest rates at payment dates:")
#for date, discount in discount_curve_schedule.items():
#    rate = getContinousInterestRate(discount, T=yearfrac(discount_curve_schedule.index[0], date, DC_CONV["INTERP"]))
#    #print(f"Date: {date.date()}, Discount Factor: {discount:.6f}, Continuous Rate: {rate:.6%}") 
#
#plt.figure(figsize=(10, 5))
#plt.plot(discount_curve_schedule.index[:15], discount_curve_schedule.values[:15], marker="o", linestyle="-")
#plt.title("Discount Curve")
#plt.xlabel("Date")
#plt.ylabel("Discount Factor")
#plt.grid(True)
#plt.tight_layout()
#plt.savefig("discount_curve.png")