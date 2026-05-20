import numpy as np
import pandas as pd
from FinDates.daycount import yearfrac

depo_converter = lambda x: float(x)/100
future_converter = lambda x: 1-float(x)/100                    
swap_converter = lambda x: float(x)/100                    

DC_CONV = {"DEPO": "ACT/360"
    , "FUTURE": "ACT/360"
    , "SWAP" : "30E/360"
    , "BOND" : "30E/360"
    , "INTERP" : "ACT/365 FIXED"}

def getZeroRates(dates, df):
    # dates : yearfractions(t0,ti) list
    # df : discount factor B(t0,ti) list
    # returns list of zero rates on ti
    
    effDates, effDf = dates, df
    if isinstance(effDates, pd.DatetimeIndex):
        effDates = list(yearfrac(list(dates), DC_CONV["INTERP"]))[1:]
        effDf = df[1:]
        print(list(-np.log(effDf) / effDates))
    return(list(-np.log(effDf) / effDates))

def getRatesLinInterpDiscount(dtSettle, dtRef, xDates, xDf, daycount=DC_CONV["INTERP"]):
    # xDates, xDf : available set of dates and discounts on which interpolate
    # dfRef: target day
    # returns discount on dtRef date
    
    assert(len(xDates) == len(xDf))
    yearFracDates = [yearfrac(dtSettle, d, daycount) for d in xDates] 
    xRates = getZeroRates(yearFracDates[1:], xDf[1:])
    yearFracRef = yearfrac(dtSettle, dtRef, daycount)
    rate = np.interp(yearFracRef, yearFracDates[1:], xRates)
    return np.exp(-rate * yearFracRef)



def bootstrapDepo(dtSettle, df_depo, df_futures, termDates, discounts):
    
    # select the correct depos and their rates
    # convert rate L(t0,ti) to discount B(t0,ti) and append the results to the current list of dates and discounts
    Idepo = (df_depo.index >= df_futures.index[0]).sum()

    depoSelected = df_depo[:Idepo].copy()
    depoDates = depoSelected.index

    depoYearFrac = [yearfrac(dtSettle, d, DC_CONV["DEPO"]) for d in depoDates]
    depoMidRates = np.mean(depoSelected[["BID", "ASK"]], axis=1).values
    depoDiscounts = [1./(1.+r*yf) for r, yf in zip(depoMidRates, depoYearFrac)]
    termDates += list(depoDates)
    discounts += depoDiscounts
    
    return termDates, discounts


def bootstrapFuture(dtSettle, df_futures, termDates, discounts):
    iFutures = 7
    # select the correct futures 
    futuresSelected = df_futures[:iFutures].copy()
    futuresYearFrac = [yearfrac(rowFut.Settle, rowFut.Expiry, DC_CONV["FUTURE"]) for t, rowFut in futuresSelected.iterrows()]
    futuresMidRates = np.mean(futuresSelected[["BID", "ASK"]], axis=1).values
    futuresSelected["Fwd"] = (1./(1.+futuresYearFrac * futuresMidRates))
    
    for t, rowFut in futuresSelected.iterrows(): 
        
        startDisc = getRatesLinInterpDiscount(dtSettle, rowFut.Settle, termDates, discounts)

        termDates += [rowFut.Expiry]
        discounts += [rowFut.Fwd * startDisc]  

    return termDates, discounts

def bootstrapSwap(dtSettle, df_swaps, termDates, discounts):
    iSwaps = (df_swaps.index >= termDates[-1]).sum()
    # select the correct swaps 
    swapsSelected = df_swaps[-iSwaps:].copy()
    swapsSelected["Mid"] = np.mean(swapsSelected[["BID", "ASK"]], axis=1).values
    
    prevSwapDate = df_swaps.index[0] 
    swapYearFrac = [yearfrac(dtSettle, prevSwapDate, DC_CONV["SWAP"])]    # list of yearfractions needed for bpv, for now yearfrac(t0,t1)
    swapDisc = [getRatesLinInterpDiscount(dtSettle, prevSwapDate, termDates, discounts)]                                              # compute the correct B(t0,t1)
    
    # recall that sn = (1 - B(t0,tn))/ BPVn  and BPVn = BPV(n-1) + yearfrac(tn-1,tn) * B(t0,tn)
    
    for swapDate, rowSwap in swapsSelected.iterrows():
        rate = rowSwap.Mid
        yf = yearfrac(prevSwapDate, swapDate, DC_CONV["SWAP"])              
        bpv = np.sum(np.multiply(swapYearFrac, swapDisc))
        df = (1 - rate * bpv) / (1 + rate * yf)
        termDates += [swapDate]
        discounts.append(df)
        swapDisc.append(df)
        swapYearFrac.append(yf)
        prevSwapDate = swapDate 
        
    return termDates, discounts

