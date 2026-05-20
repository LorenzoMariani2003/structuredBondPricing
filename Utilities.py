import sys, os
import datetime as dt
#from typing import final
import numpy as np
import pandas as pd
import FinDates.holidays as holidays
import FinDates.busdayrule as busdayrule 
from dateutil.relativedelta import relativedelta 

numberDaysYear = 256


class Calendar:
    
    PeriodUnit = 'Unit'
    PeriodLabel = 'Label'

    def __init__(self, calendarCode="de.eurex"):
        self.Holidays = holidays.get_calendar(calendarCode)
    
    def is_holiday(self, refDate):
        return self.Holidays.is_holiday(refDate)

    @staticmethod
    def periodParser(period):
        idx, out = 0, {}
        for i in range(len(period)):
            if period[i].isalpha():
                idx = i
                break
        out[Calendar.PeriodUnit] = int(period[:idx])
        out[Calendar.PeriodLabel] = period[idx:].upper()
        return out

    def __dateAddNumberOfDays__(self, startDate, businessDays):
        endDate = pd.to_datetime(startDate)
        direction, iSize = np.sign(businessDays), np.abs(businessDays)
        for _ in range(iSize):
            if direction<0. : endDate -= dt.timedelta(1)
            else : endDate += dt.timedelta(1)
            while self.is_holiday(endDate):
                if direction<0. : endDate -= dt.timedelta(1)
                else : endDate += dt.timedelta(1)
        return pd.to_datetime(endDate)

    def __dateAddNumberOfMonths__(self, startDate, numberOfMonths):
        endDate = pd.to_datetime(startDate)
        return (endDate + relativedelta(months=numberOfMonths))
        
    def dateAdjust(self, startDate, adjustmentRule):
        return busdayrule.rolldate(startDate, self.Holidays, adjustmentRule)

    def dateAddPeriod(self, startDate, period, adjustmentRule):
        parserResult, endDate = Calendar.periodParser(period), dt.datetime.min
        periodUnit, periodLabel = parserResult[Calendar.PeriodUnit], parserResult[Calendar.PeriodLabel]
        if(periodLabel == 'BD'): endDate = self.__dateAddNumberOfDays__(startDate, periodUnit)
        elif(periodLabel == 'W'): endDate = self.__dateAddNumberOfDays__(startDate, periodUnit*5)
        elif(periodLabel == 'M'): endDate = self.__dateAddNumberOfMonths__(startDate, periodUnit)
        elif(periodLabel == 'Y'): endDate = self.__dateAddNumberOfMonths__(startDate, periodUnit*12)
        else: raise ValueError("Invalid period {}".format(period))
        return self.dateAdjust(endDate, adjustmentRule)


def createSchedule(refDate, periods, calendarCode="de.eurex", adjustmentRule="modfollow"):
    schedule = []
    for period in periods:
        schedule += [Calendar(calendarCode).dateAddPeriod(refDate, period, adjustmentRule).date()]
    return schedule
