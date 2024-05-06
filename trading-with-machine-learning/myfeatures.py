############################################################################################################
###--- This script contains different sets of features - financial indicators grouped in a dataframe, ---###
###--- that will be used as input in machine learning models, etc. --------------------------------------###
############################################################################################################

from technical_indicators import *
import talib
import myindicators
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from salib import *
from datetime import datetime, timedelta


def GetFeatures(feature_name,rates,params=None,pTimeFrame=None):
    """This function returns a dataframe with a set of indicators (features) that will be input for the machine learning models"""

    if feature_name == "FeatureD1": return GetFeaturesD1(rates)               # features for daily candles
    elif feature_name == "FeatureNewton": return GetFeaturesNewton(rates)  # features using newton's paper indicators + hour + gap
    elif feature_name == "FeaturesSALIB_THX": return GetFeaturesSALIB_THX(rates,params)     # features using the salib and technical indicators



def GetFeaturesNewton(rates):
    """Create the features similarly to Newton's paper"""

    df = pd.DataFrame()
    fhigh4, flow4, fhigh0, flow0, fopen0 = myindicators.newton_features(rates)  # calculates the newton's paper features
    # time = [rates[i]["time"].hour * 60 + rates[i]["time"].minute for i, item in enumerate(rates)]
    time = [rates[i]["time"].hour for i, item in enumerate(rates)]

    df["fhigh4"] = np.array(fhigh4)     # percentage change between the max of 4 days and the current close price
    df["flow4"] = np.array(flow4)       # percentage change between the min of 4 days and the current close price
    df["fhigh0"] = np.array(fhigh0)     # percentage change between high and close of the current day
    df["flow0"] = np.array(flow0)       # percentage change between low and close of the current day
    df["fopen0"] = np.array(fopen0)     # percentage change between open and close of the current day
    # df["c"] = np.array([rates[i]["close"] for i, item in enumerate(rates)])   # close prices
    df["time"] = np.array(time)         # hour of the day
    df["gap"] = np.array(myindicators.getGaps(rates))   # gap between today's open and yesterday's close prices

    # #--- to visualize the indicators
    # df["fopen0"].plot()
    # plt.show()

    return df



def GetFeaturesD1(rates):
    """A few features to be used with daily candles"""
    df = pd.DataFrame()

    df["VAR-0"] = [(rates[max(0, i-0)]["close"] - rates[max(0, i-0)]["open"]) / rates[max(0, i-0)]["open"] for i, item in enumerate(rates)]            # today's percentahe change
    df["VAR-1"] = [(rates[max(0, i-1)]["close"] - rates[max(0, i-1)]["open"]) / rates[max(0, i-1)]["open"] for i, item in enumerate(rates)]            # yesterday's percentahe change
    df["VAR-2"] = [(rates[max(0, i-2)]["close"] - rates[max(0, i-2)]["open"]) / rates[max(0, i-2)]["open"] for i, item in enumerate(rates)]            # ...
    df["GAP-0"] = [(rates[max(0, i-0)]["close"] - rates[max(0, min(i+1, len(rates)-1))]["open"]) / rates[max(0, i-0)]["close"] for i, item in enumerate(rates)]  # gap between the close of the current candle and the open of the next one.
                                                                                                                                                                        # Notice we are not seeing the future in this case, because although we think                                                                                                                                                          # que opening of the next candle. Thus we can use the gap+1.
    df["CTR"] = [(rates[max(0, i-0)]["close"] - rates[max(0, i-0)]["low"]) / (rates[max(0, i-0)]["high"] - rates[max(0, i-0)]["low"]+5) for i, item in enumerate(rates)]

    df["lowVAR-0"] = [(rates[max(0, i - 0)]["low"] - rates[max(0, i - 0)]["open"]) / rates[max(0, i - 0)]["open"] for i, item in enumerate(rates)]
    df["highVAR-0"] = [(rates[max(0, i - 0)]["high"] - rates[max(0, i - 0)]["open"]) / rates[max(0, i - 0)]["open"] for i, item in enumerate(rates)]
    df["rangeVAR-0"] = [(rates[max(0, i - 0)]["high"] - rates[max(0, i - 0)]["low"]) / rates[max(0, i - 0)]["open"] for i, item in enumerate(rates)]

    return df



def GetFeaturesSALIB_THX(rates,params,mode="rates"):
    """Features using the salib and technical indicators"""

    df = pd.DataFrame()
    print("Calculating indicators!!")

    # --- separating some useful variables in order to use them for calculating some features
    if mode == "rates":
        time = np.array([rate["time"] for rate in rates])
        open_ = np.array([rate["open"] for rate in rates])
        high = np.array([rate["high"] for rate in rates])
        low = np.array([rate["low"] for rate in rates])
        close = np.array([rate["close"] for rate in rates])
        volume = np.array([float(rate["volume"]) for rate in rates])
    else: # if we are using an pd dataframe instead o dictionary of dictionaries (rates)
        time = np.array(rates["time"])
        open_ = np.array(rates["open"])
        high = np.array(rates["high"])
        low = np.array(rates["low"])
        close = np.array(rates["close"])
        volume = np.array(rates["volume"])


    # --- Getting daily, weekly and monthly timeframes
    open_day, high_day, low_day, close_day = transform_rates(time, open_, high, low, close, timeframe="day", n=34)
    open_week, high_week, low_week, close_week = transform_rates(time, open_, high, low, close, timeframe="week", n=5)
    open_month, high_month, low_month, close_month = transform_rates(time, open_, high, low, close, timeframe="month", n=1)
    open_h1, high_h1, low_h1, close_h1 = transform_rates(time, open_, high, low, close, timeframe="h1", n=17)
    open_m20, high_m20, low_m20, close_m20 = transform_rates(time, open_, high, low, close, timeframe="m20", n=5)
    open_m15, high_m15, low_m15, close_m15 = transform_rates(time, open_, high, low, close, timeframe="m15", n=17)
    # df["close"] = close

    if 'open' in params:
        df["open"] = open_
    if 'high' in params:
        df["high"] = high
    if 'low' in params:
        df["low"] = low
    if 'close' in params:
        df["close"] = close

    #--- CTO
    if 'cto1' in params:
        cto1 = change(close_day[1], open_day[1], "percent")   # Close to Open
        df["cto1"] = cto1
    if 'cto2' in params:
        cto2 = change(close_day[2], open_day[2], "percent")   # Close to Open
        df["cto2"] = cto2
    if 'cto3' in params:
        cto3 = change(close_day[3], open_day[3], "percent")   # Close to Open
        df["cto3"] = cto3


    if 'cto1_h1' in params:
        cto1_h1 = change(close_h1[1], open_h1[1], "percent")   # Close to Open
        df["cto1_h1"] = cto1_h1
    if 'cto2_h1' in params:
        cto2_h1 = change(close_h1[2], open_h1[2], "percent")   # Close to Open
        df["cto2_h1"] = cto2_h1
    if 'cto2_h1' in params:
        cto3_h1 = change(close_h1[3], open_h1[3], "percent")   # Close to Open
        df["cto3_h1"] = cto3_h1


    # add 2022-09-28
    if 'ctc0' in params:
        ctc0 = change(close_day[0], close_day[1], "direction")  # Close to Close
        df["ctc0"] = ctc0
    if 'ctc1' in params:
        ctc1 = change(close_day[1], close_day[2], "direction")  # Close to Close
        df["ctc1"] = ctc1
    if 'ctc2' in params:
        ctc2 = change(close_day[2], close_day[3], "direction")  # Close to Close
        df["ctc2"] = ctc2
    if 'ctc3' in params:
        ctc3 = change(close_day[3], close_day[4], "direction")  # Close to Close
        df["ctc3"] = ctc3
    if 'ctc4' in params:
        ctc4 = change(close_day[4], close_day[5], "direction")  # Close to Close
        df["ctc4"] = ctc4
    if 'ctc5' in params:
        ctc5 = change(close_day[5], close_day[6], "direction")  # Close to Close
        df["ctc5"] = ctc5


    if 'ctc0_label' in params:
        ctc0_label = label(change(close_day[0], close_day[1], "direction"),[0])  # direction of CTC (above or below 0)
        df["ctc0_label"] = ctc0_label
    if 'ctc1_label' in params:
        ctc1_label = label(change(close_day[1], close_day[2], "direction"),[0])  # direction of CTC (above or below 0)
        df["ctc1_label"] = ctc1_label
    if 'ctc2_label' in params:
        ctc2_label = label(change(close_day[2], close_day[3], "direction"),[0])  # direction of CTC (above or below 0)
        df["ctc2_label"] = ctc2_label
    if 'ctc3_label' in params:
        ctc3_label = label(change(close_day[3], close_day[4], "direction"),[0])  # direction of CTC (above or below 0)
        df["ctc3_label"] = ctc3_label
    if 'ctc4_label' in params:
        ctc4_label = label(change(close_day[4], close_day[5], "direction"),[0])  # direction of CTC (above or below 0)
        df["ctc4_label"] = ctc4_label
    if 'ctc5_label' in params:
        ctc5_label = label(change(close_day[5], close_day[6], "direction"),[0])  # direction of CTC (above or below 0)
        df["ctc5_label"] = ctc5_label


    if 'ctc0_month' in params:
        ctc0_month = change(close_month[0], close_month[1], "direction")  # CTC
        df["ctc0_month"] = ctc0_month
    if 'ctc1_month' in params:
        ctc1_month = change(close_month[1], close_month[2],"direction")  # CTC
        df["ctc1_month"] = ctc1_month
    if 'ctc2_month' in params:
        ctc2_month = change(close_month[2], close_month[3],"direction")  # CTC
        df["ctc2_month"] = ctc2_month


    if 'cto0_month' in params:
        cto0_month = change(close_month[0], close_month[1], "direction")  # CTO
        df["cto0_month"] = cto0_month
    if 'cto1_month' in params:
        cto1_month = change(close_month[1], close_month[2],"direction")  # CTO
        df["cto1_month"] = cto1_month
    if 'cto2_month' in params:
        cto2_month = change(close_month[2], close_month[3],"direction")  # CTO
        df["cto2_month"] = cto2_month


    if 'fopen0' in params:
        fopen0 = change(close, open_day[0], "percent")  # close from daily open (CTO)
        df["fopen0"] = fopen0
    if 'fhigh0' in params:
        fhigh0 = change(close, high_day[0], "percent")  # close from daily high
        df["fhigh0"] = fhigh0
    if 'flow0' in params:
        flow0 = change(close, low_day[0], "percent")    # close from daily low
        df["flow0"] = flow0
    if 'fhigh4' in params:
        fhigh4 = change(close, cmax(high_day[0:5]), "percent")  # close from high.n
        df["fhigh4"] = fhigh4
    if 'flow4' in params:
        flow4 = change(close, cmin(low_day[0:5]), "percent")    # close from low.n
        df["flow4"] = flow4
    if 'fhigh8' in params:
        fhigh8 = change(close, cmax(high_day[0:9]), "percent")  # close from high.n
        df["fhigh8"] = fhigh8
    if 'flow8' in params:
        flow8 = change(close, cmin(low_day[0:9]), "percent")    # close from low.n
        df["flow8"] = flow8
    if 'fhigh12' in params:
        fhigh12 = change(close, cmax(high_day[0:13]), "percent")  # close from high.n
        df["fhigh12"] = fhigh12
    if 'flow12' in params:
        flow12 = change(close, cmin(low_day[0:13]), "percent")    # close from low.n
        df["flow12"] = flow12
    if 'fhigh10' in params:
        fhigh10 = change(close, cmax(high_day[0:11]), "percent")  # close from high.n
        df["fhigh10"] = fhigh10
    if 'flow10' in params:
        flow10 = change(close, cmin(low_day[0:11]), "percent")    # close from low.n
        df["flow10"] = flow10
    if 'fhigh20' in params:
        fhigh20 = change(close, cmax(high_day[0:21]), "percent")  # close from high.n
        df["fhigh20"] = fhigh20
    if 'flow20' in params:
        flow20 = change(close, cmin(low_day[0:21]), "percent")    # close from low.n
        df["flow20"] = flow20


    if 'fopen1' in params:
        fopen1 = change(close, open_day[1], "percent")  # close from daily open (CTO)
        df["fopen1"] = fopen1
    if 'fopen2' in params:
        fopen2 = change(close, open_day[2], "percent")  # close from daily open (CTO)
        df["fopen2"] = fopen2
    if 'fopen4' in params:
        fopen4 = change(close, open_day[4], "percent")  # close from daily open (CTO)
        df["fopen4"] = fopen4
    if 'fopen8' in params:
        fopen8 = change(close, open_day[8], "percent")  # close from daily open (CTO)
        df["fopen8"] = fopen8


    if 'fopen32' in params:
        fopen32 = change(close, open_day[32], "percent")  # close from daily open (CTO)
        df["fopen32"] = fopen32
    if 'fhigh32' in params:
        fhigh32 = change(close, cmax(high_day[0:33]), "percent")  # close from high.n
        df["fhigh32"] = fhigh32
    if 'flow32' in params:
        flow32 = change(close, cmin(low_day[0:33]), "percent")    # close from low.n
        df["flow32"] = flow32


    if 'fopen0_h1' in params:
        fopen0_h1 = change(close, open_h1[0], "percent")  # close from daily open (CTO)
        df["fopen0_h1"] = fopen0_h1
    if 'fhigh0_h1' in params:
        fhigh0_h1 = change(close, high_h1[0], "percent")  # close from daily high
        df["fhigh0_h1"] = fhigh0_h1
    if 'flow0_h1' in params:
        flow0_h1 = change(close, low_h1[0], "percent")    # close from daily low
        df["flow0_h1"] = flow0_h1
    if 'fhigh4_h1' in params:
        fhigh4_h1 = change(close, cmax(high_h1[0:5]), "percent")  # close from high.n
        df["fhigh4_h1"] = fhigh4_h1
    if 'flow4_h1' in params:
        flow4_h1 = change(close, cmin(low_h1[0:5]), "percent")    # close from low.n
        df["flow4_h1"] = flow4_h1


    if 'fopen0_m20' in params:
        fopen0_m20 = change(close, open_m20[0], "percent")  # close from daily open (CTO)
        df["fopen0_m20"] = fopen0_m20
    if 'fhigh0_m20' in params:
        fhigh0_m20 = change(close, high_m20[0], "percent")  # close from daily high
        df["fhigh0_m20"] = fhigh0_m20
    if 'flow0_m20' in params:
        flow0_m20 = change(close, low_m20[0], "percent")    # close from daily low
        df["flow0_m20"] = flow0_m20


    #--- IBS
    if 'ibs0' in params:
        ibs0 = price_to_range(close, high_day[0], low_day[0])
        df["ibs0"] = ibs0
    if 'ibs1' in params:
        ibs1 = price_to_range(close, cmax(high_day[0:2]), cmin(low_day[0:2]))
        df["ibs1"] = ibs1
    if 'ibs2' in params:
        ibs2 = price_to_range(close, cmax(high_day[0:3]), cmin(low_day[0:3]))
        df["ibs2"] = ibs2
    if 'ibs4' in params:
        ibs4 = price_to_range(close, cmax(high_day[0:5]), cmin(low_day[0:5]))
        df["ibs4"] = ibs4
    if 'ibs8' in params:
        ibs8 = price_to_range(close, cmax(high_day[0:9]), cmin(low_day[0:9]))
        df["ibs8"] = ibs8
    if 'ibs16' in params:
        ibs16 = price_to_range(close, cmax(high_day[0:17]), cmin(low_day[0:17]))
        df["ibs16"] = ibs16


    if 'ibs0_h1' in params:
        ibs0_h1 = price_to_range(close, high_h1[0], low_h1[0])
        df["ibs0_h1"] = ibs0_h1
    if 'ibs1_h1' in params:
        ibs1_h1 = price_to_range(close, cmax(high_h1[0:2]), cmin(low_h1[0:2]))
        df["ibs1_h1"] = ibs1_h1
    if 'ibs2_h1' in params:
        ibs2_h1 = price_to_range(close, cmax(high_h1[0:3]), cmin(low_h1[0:3]))
        df["ibs2_h1"] = ibs2_h1
    if 'ibs4_h1' in params:
        ibs4_h1 = price_to_range(close, cmax(high_h1[0:5]), cmin(low_h1[0:5]))
        df["ibs4_h1"] = ibs4_h1
    if 'ibs8_h1' in params:
        ibs8_h1 = price_to_range(close, cmax(high_h1[0:9]), cmin(low_h1[0:9]))
        df["ibs8_h1"] = ibs8_h1


    if 'range0_h1' in params:
        range0_h1 = change(high_h1[0], low_h1[0], "percent")  # range h1
        df["range0_h1"] = range0_h1
    if 'range1_h1' in params:
        range1_h1 = change(cmax(high_h1[0:2]), cmin(low_h1[0:2]), "percent")  # range h1
        df["range1_h1"] = range1_h1
    if 'range2_h1' in params:
        range2_h1 = change(cmax(high_h1[0:3]), cmin(low_h1[0:3]), "percent")  # range h1
        df["range2_h1"] = range2_h1
    if 'range4_h1' in params:
        range4_h1 = change(cmax(high_day[0:5]), cmin(low_day[0:5]), "percent")  # range h1
        df["range4_h1"] = range4_h1


    if 'ibs0c' in params:
        ibs0c = price_to_range(close, high_m15[0], low_m15[0])
        df["ibs0c"] = ibs0c
    if 'ibs1c' in params:
        ibs1c = price_to_range(close, cmax(high_m15[0:2]), cmin(low_m15[0:2]))
        df["ibs1c"] = ibs1c
    if 'ibs4c' in params:
        ibs4c = price_to_range(close, cmax(high_m15[0:5]), cmin(low_m15[0:5]))
        df["ibs4c"] = ibs4c
    if 'ibs8c' in params:
        ibs8c = price_to_range(close, cmax(high_m15[0:9]), cmin(low_m15[0:9]))
        df["ibs8c"] = ibs8c
    if 'ibs16c' in params:
        ibs16c = price_to_range(close, cmax(high_m15[0:17]), cmin(low_m15[0:17]))
        df["ibs16c"] = ibs16c


    if 'range0c' in params:
        range0c = change(high_m15[0], low_m15[0], "percent")     # daily m15
        df["range0c"] = range0c
    if 'range1c' in params:
        range1c = change(high_m15[1], low_m15[1], "percent")  # daily m15
        df["range1c"] = range1c
    if 'range2c' in params:
        range2c = change(cmax(high_m15[0:3]), cmin(low_m15[0:3]), "percent")  # daily m15
        df["range2c"] = range2c
    if 'range4c' in params:
        range4c = change(cmax(high_m15[0:5]), cmin(low_m15[0:5]), "percent")  # daily m15
        df["range4c"] = range4c
    if 'range8c' in params:
        range8c = change(cmax(high_m15[0:9]), cmin(low_m15[0:9]), "percent")  # daily m15
        df["range8c"] = range8c
    if 'range16c' in params:
        range16c = change(cmax(high_m15[0:17]), cmin(low_m15[0:17]), "percent")  # daily m15
        df["range16c"] = range16c


    if 'gap' in params:
        gap = change(open_day[0],close_day[1],"percent")
        df["gap"] = gap


    if 'range0' in params:
        range0 = change(high_day[0], low_day[0], "percent")     # daily range
        df["range0"] = range0
    if 'range1' in params:
        range1 = change(high_day[1], low_day[1], "percent")  # daily range
        df["range1"] = range1
    if 'range2' in params:
        range2 = change(cmax(high_day[0:3]), cmin(low_day[0:3]), "percent")  # daily range
        df["range2"] = range2
    if 'range4' in params:
        range4 = change(cmax(high_day[0:5]), cmin(low_day[0:5]), "percent")  # daily range
        df["range4"] = range4
    if 'range8' in params:
        range8 = change(cmax(high_day[0:9]), cmin(low_day[0:9]), "percent")  # daily range
        df["range8"] = range8
    if 'range16' in params:
        range16 = change(cmax(high_day[0:17]), cmin(low_day[0:17]), "percent")  # daily range
        df["range16"] = range16


    if 'range_week0' in params:
        range_week0 = change(high_week[0],low_week[0],"percent")    # weekly range
        df["range_week0"] = range_week0
    if 'range_week1' in params:
        range_week1 = change(cmax(high_week[0:2]), cmin(low_week[0:2]), "percent")    # weekly range
        df["range_week1"] = range_week1
    if 'range_month0' in params:
        range_month0 = change(high_month[0], low_month[0], "percent")    # monthly range
        df["range_month0"] = range_month0


    if 'average_range_10' in params:
        average_range_10 = mean([change(high_day[i], low_day[i], "percent") for i in range(1,10)])     # average "n" days range
        df["average_range_10"] = average_range_10
    if 'average_range_20' in params:
        average_range_20 = mean([change(high_day[i], low_day[i], "percent") for i in range(1, 20)])  # average "n" days range
        df["average_range_20"] = average_range_20
    if 'average_range_100' in params:
        average_range_100 = mean([change(high_day[i], low_day[i], "percent") for i in range(1, 100)])  # average "n" days range
        df["average_range_100"] = average_range_100
    if 'average_range_5' in params:
        average_range_5 = mean([change(high_day[i], low_day[i], "percent") for i in range(1, 5)])  # average "n" days range
        df["average_range_5"] = average_range_5


    if 'average_failure_high' in params:
        average_failure_high = mean([failure_high(high_day[i], open_day[i], close_day[i], "percent") for i in range(1,30)], exception=-1)
        df["average_failure_high"] = average_failure_high
    if 'average_failure_low' in params:
        average_failure_low = mean([failure_low(low_day[i], open_day[i], close_day[i], "percent") for i in range(1, 30)], exception=-1)
        df["average_failure_low"] = average_failure_low


    from salib import failure_size
    if 'average_failure_size_day_15' in params:
        average_failure_size = mean([failure_size( open_day[i], high_day[i], low_day[i], "percent") for i in range(1,15)])
        df["average_failure_size_day_15"] = average_failure_size


    if 'average_failure_size_h1_10' in params:
        average_failure_size = mean([failure_size(open_h1[i], high_h1[i], low_h1[i], "percent") for i in range(1,10)])
        df["average_failure_size_h1_10"] = average_failure_size


    if 'failure_size_day' in params:
        failure_size = failure_size(open_day[0], high_day[0], low_day[0], "percent")
        df["failure_size_day"] = failure_size


    #--- Daily position
    if 'daily_position0' in params:
        daily_position0 = daily_position(high_day[0], low_day[0], high_day[1], low_day[1])
        df["daily_position0"] = daily_position0
    if 'daily_position1' in params:
        daily_position1 = daily_position(high_day[1], low_day[1], high_day[2], low_day[2])
        df["daily_position1"] = daily_position1
    if 'daily_position2' in params:
        daily_position2 = daily_position(high_day[2], low_day[2], high_day[3], low_day[3])
        df["daily_position2 "] = daily_position2
    if 'daily_position3' in params:
        daily_position3 = daily_position(high_day[3], low_day[3], high_day[4], low_day[4])
        df["daily_position3 "] = daily_position3
    if 'weekly_position0' in params:
        weekly_position0 = daily_position(high_week[0], low_week[0], high_week[1], low_week[1])
        df["weekly_position0"] = weekly_position0
    if 'weekly_position1' in params:
        weekly_position1 = daily_position(high_week[1], low_week[1], high_week[2], low_week[2])
        df["weekly_position1"] = weekly_position1
    if 'weekly_position2' in params:
        weekly_position2 = daily_position(high_week[2], low_week[2], high_week[3], low_week[3])
        df["weekly_position2"] = weekly_position2


    if 'relative_position0' in params:
        relative_position0 = relative_position(close, high_day[0], low_day[0])
        df["relative_position0"] = relative_position0


    if 'minute' in params:
        minute = [time[i].minute for i, item in enumerate(time)]
        df["minute"] = minute
    if 'minutes' in params:
        minutes = [time[i].hour*60 + time[i].minute for i, item in enumerate(time)]
        df["minutes"] = minutes
    if 'hour' in params:
        hour = [time[i].hour for i, item in enumerate(time)]
        df["hour"] = hour
    if 'day' in params:
        day = [time[i].day for i, item in enumerate(time)]
        df["day"] = day
    if 'weekday' in params:
        day = [time[i].weekday() for i, item in enumerate(time)]
        df["weekday"] = day
    if 'day_label' in params:
        day_label = label([time[i].day for i, item in enumerate(time)],[10,20])
        df["day_label"] = day_label
    if 'quadrant' in params:
        quadrant = month_quadrant(time, open_month[0], close_month[0])
        df["quadrant"] = quadrant


    if 'cto1_label' in params:
        cto1_label = label(change(close_day[1], open_day[1], "direction"),[0])  # direction (above or below 0) of close.1-open.1
        df["cto1_label"] = cto1_label
    if 'cto2_label' in params:
        cto2_label = label(change(close_day[2], open_day[2], "direction"),[0])  # direction (above or below 0) of close.2-open.2
        df["cto2_label"] = cto2_label
    if 'cto3_label' in params:
        cto3_label = label(change(close_day[3], open_day[3], "direction"),[0])  # direction (above or below 0) of close.3-open.3
        df["cto3_label"] = cto3_label
    if 'cto4_label' in params:
        cto4_label = label(change(close_day[4], open_day[4], "direction"),[0])  # direction (above or below 0) of close.3-open.3
        df["cto4_label"] = cto4_label


    if 'cto1_label2' in params:
        cto1_label2 = label(change(close_day[1], open_day[1], "percent"),[-0.01,-0.005,0,0.005,0.01])  # direction of CTO (above or below 0)
        df["cto1_label2"] = cto1_label2

    if 'gap_label2' in params:
        gap_label2 = label(change(open_day[0],close_day[1],"percent"),[-0.01,-0.005,0,0.005,0.01])  # direction of GAP (above or below 0)
        df["gap_label2"] = gap_label2

    if 'ibs1_label2' in params:
        ibs1_label2 = label(price_to_range(close, cmax(high_day[0:2]), cmin(low_day[0:2])),[0.3,0.7])
        df["ibs1_label2"] = ibs1_label2


    if 'cto1_label2_i' in params:
        cto1_label2_i = label(change(close_day[1], open_day[1], "percent"),[-0.01,-0.005,0,0.005,0.01])  # CTO divided in labels
        cto1_label2_i = [6-x for x in cto1_label2_i]
        df["cto1_label2_i"] = cto1_label2_i

    if 'gap_label2_i' in params:
        gap_label2_i = label(change(open_day[0],close_day[1],"percent"),[-0.01,-0.005,0,0.005,0.01])  # Gap divided in labels
        gap_label2_i = [6 - x for x in gap_label2_i]
        df["gap_label2_i"] = gap_label2_i

    if 'ibs1_label2_i' in params:
        ibs1_label2_i = label(price_to_range(close, cmax(high_day[0:2]), cmin(low_day[0:2])),[0.3,0.7])
        ibs1_label2_i = [3 - x for x in ibs1_label2_i]
        df["ibs1_label2_i"] = ibs1_label2_i

    if 'cto1_label2_e' in params:
        cto1_label2_e = label(change(close_day[1], open_day[1], "percent"),[-0.01,-0.005,0,0.005,0.01])
        df["cto1_label2_e0"] = [(1 if x==0 else 0) for x in cto1_label2_e]
        df["cto1_label2_e1"] = [(1 if x == 1 else 0) for x in cto1_label2_e]
        df["cto1_label2_e2"] = [(1 if x == 2 else 0) for x in cto1_label2_e]
        df["cto1_label2_e3"] = [(1 if x == 3 else 0) for x in cto1_label2_e]
        df["cto1_label2_e4"] = [(1 if x == 4 else 0) for x in cto1_label2_e]
        df["cto1_label2_e5"] = [(1 if x == 5 else 0) for x in cto1_label2_e]

    if 'gap_label2_e' in params:
        gap_label2_e = label(change(open_day[0],close_day[1],"percent"),[-0.01,-0.005,0,0.005,0.01])
        df["gap_label2_e0"] = [(1 if x==0 else 0) for x in gap_label2_e]
        df["gap_label2_e1"] = [(1 if x == 1 else 0) for x in gap_label2_e]
        df["gap_label2_e2"] = [(1 if x == 2 else 0) for x in gap_label2_e]
        df["gap_label2_e3"] = [(1 if x == 3 else 0) for x in gap_label2_e]
        df["gap_label2_e4"] = [(1 if x == 4 else 0) for x in gap_label2_e]
        df["gap_label2_e5"] = [(1 if x == 5 else 0) for x in gap_label2_e]

    if 'ibs1_label2_e' in params:
        ibs1_label2_e = label(price_to_range(close, cmax(high_day[0:2]), cmin(low_day[0:2])),[0.3,0.7])  # direction (above or below 0) of close.1-open.1
        df["ibs1_label2_e0"] = [(1 if x==0 else 0) for x in ibs1_label2_e]
        df["ibs1_label2_e1"] = [(1 if x == 1 else 0) for x in ibs1_label2_e]
        df["ibs1_label2_e2"] = [(1 if x == 2 else 0) for x in ibs1_label2_e]


    if 'candle_id' in params:
        df['candle_id'] = myindicators.getCandleId(rates)  # get the id of the candle - if it is the first candle of the day, or second ...


    # --- getting labeled features based on technical indicators
    if 'ma_9_40_label' in params:
        df["ma_9_40_label"], _, _ = myindicators.movingAverageLabel(close, period_fast=9, period_slow=40)

    if 'stoch_4_label' in params:
        df["stoch_4_label"], _, _ = myindicators.stochLabel(high, low, close, fastk_period=4, slowk_period=3,
                                                                slowk_matype=0, slowd_period=3, slowd_matype=0)
    if 'stoch_14_label' in params:
        df["stoch_14_label"], _, _ = myindicators.stochLabel(high, low, close, fastk_period=14, slowk_period=3,
                                                                slowk_matype=0, slowd_period=3, slowd_matype=0)
    if 'stoch_32_label' in params:
        df["stoch_32_label"], _, _ = myindicators.stochLabel(high, low, close, fastk_period=32, slowk_period=3,
                                                                  slowk_matype=0, slowd_period=3, slowd_matype=0)
    if 'stoch_4' in params:
        _, df["stoch_4"], _ = myindicators.stochLabel(high, low, close, fastk_period=4, slowk_period=3,
                                                                slowk_matype=0, slowd_period=3, slowd_matype=0)
    if 'stoch_14' in params:
        _, df["stoch_14"], _ = myindicators.stochLabel(high, low, close, fastk_period=14, slowk_period=3,
                                                                slowk_matype=0, slowd_period=3, slowd_matype=0)
    if 'stoch_32' in params:
        _, df["stoch_32"], _ = myindicators.stochLabel(high, low, close, fastk_period=32, slowk_period=3,
                                                                  slowk_matype=0, slowd_period=3, slowd_matype=0)

    if 'adx_14' in params:
        df["adx_14"] = talib.ADX(high, low, close, timeperiod=14)
    if 'adx_32' in params:
        df["adx_32"] = talib.ADX(high, low, close, timeperiod=32)
    if 'adx_64' in params:
        df["adx_64"] = talib.ADX(high, low, close, timeperiod=64)
    if 'rsi_4' in params:
        df["rsi_4"] = talib.RSI(close, timeperiod=4)
    if 'rsi_14' in params:
        df["rsi_14"] = talib.RSI(close, timeperiod=14)
    if 'rsi_42' in params:
        df["rsi_42"] = talib.RSI(close, timeperiod=42)

    if 'ma' in params:
        df["ma"] = talib.MA(close, timeperiod=20, matype=0) - talib.MA(close, timeperiod=2, matype=0)
    if 'ma_21_2' in params:
        df["ma_21_2"] = talib.MA(close, timeperiod=21, matype=0) / talib.MA(close, timeperiod=2, matype=0)
    if 'ma_100_10' in params:
        df["ma_100_10"] = talib.MA(close, timeperiod=100, matype=0) / talib.MA(close, timeperiod=10, matype=0)
    if 'range_1' in params:
        df["range_1"] = talib.ATR(high, low, close, timeperiod=9)
    if 'range_2' in params:
        df["range_2"] = talib.ATR(high, low, close, timeperiod=21)

    if 'candle_return_1' in params:
        df["candle_return_1"] = [(rates[i]["close"] - rates[i]["open"]) / rates[i]["open"] for i, item in
                                 enumerate(rates)]  # get the percentage change of the last candle

    if 'candle_return_2' in params:
        df["candle_return_2"] = [(rates[i]["close"] - rates[max(0, i - 10)]["open"]) / rates[i]["open"] for i, item in
                                 enumerate(rates)]  # get the percentage change of the last candle

    if 'close_price' in params:
        df["close_price"] = np.array([rates[i]["close"] for i, item in enumerate(rates)])   # close prices


    if 'atr_10' in params:
        df["atr_10"] = talib.ATR(high, low, close, timeperiod=10)

    if 'natr_4' in params:
        df["natr_4"] = talib.NATR(high, low, close, timeperiod=4)

    if 'natr_14' in params:
        df["natr_14"] = talib.NATR(high, low, close, timeperiod=14)

    if 'atr_4' in params:
        df["atr_4"] = talib.NATR(high, low, close, timeperiod=4)

    if 'atr_14' in params:
        df["atr_14"] = talib.NATR(high, low, close, timeperiod=14)


    if "bbands_4_label" in params:
        df["bbands_4_label"], _, _, _ = myindicators.bbandsLabel(close, timeperiod=4, nbdevup=2, nbdevdn=2, matype=0)

    if "bbands_14_label" in params:
        df["bbands_14_label"], _, _, _ = myindicators.bbandsLabel(close, timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)

    if "bbands_48_label" in params:
        df["bbands_48_label"], _, _, _ = myindicators.bbandsLabel(close, timeperiod=48, nbdevup=2, nbdevdn=2, matype=0)

    if "pivot_label"in params:
        df["pivot_label"] = myindicators.pivotLabel(rates)

    if  "pivot_ibs3" in params or "pivot_ibs2" in params or "pivot_ibs1" in params:
        from myindicators import getPivotPoint
        p, r1, r2, r3, s1, s2, s3 = getPivotPoint(high_day[1], low_day[1], close_day[1])
        if "pivot_ibs1" in params:
            df["pivot_ibs1"] = price_to_range(close, r1, s1)
        if "pivot_ibs2" in params:
            df["pivot_ibs2"] = price_to_range(close, r2, s2)
        if "pivot_ibs3" in params:
            df["pivot_ibs3"] = price_to_range(close, r3, s3)


    if  "pivot_from_p" in params or "pivot_from_r1" in params or "pivot_from_s1" in params or \
            "pivot_from_r2" in params or "pivot_from_s2" in params or \
            "pivot_from_r3" in params or "pivot_from_s3" in params:
        from myindicators import getPivotPoint
        p, r1, r2, r3, s1, s2, s3 = getPivotPoint(high_day[1], low_day[1], close_day[1])
        if "pivot_from_p" in params:
            df["pivot_from_p"] = change(close, p, "percent")
        if "pivot_from_r1" in params:
            df["pivot_from_r1"] = change(close, r1, "percent")
        if "pivot_from_s1" in params:
            df["pivot_from_s1"] = change(close, s1, "percent")
        if "pivot_from_r2" in params:
            df["pivot_from_r2"] = change(close, r2, "percent")
        if "pivot_from_s2" in params:
            df["pivot_from_s2"] = change(close, s2, "percent")
        if "pivot_from_r3" in params:
            df["pivot_from_r3"] = change(close, r3, "percent")
        if "pivot_from_s3" in params:
            df["pivot_from_s3"] = change(close, s3, "percent")


    if 'volume1_5_10' in params:
        volume_ma_fast = talib.MA(volume, timeperiod=5, matype=0)
        volume_ma_slow = talib.MA(volume, timeperiod=10, matype=0)
        volume_diff = []
        for i in range(0,len(volume_ma_fast)):
            if volume_ma_fast[max(0,i-1)]<volume_ma_slow[max(0,i-1)] and volume_ma_fast[i]>=volume_ma_slow[i]:
                volume_diff.append(2)
            elif volume_ma_fast[max(0,i-1)]>volume_ma_slow[max(0,i-1)] and volume_ma_fast[i]<=volume_ma_slow[i]:
                volume_diff.append(4)
            elif volume_ma_fast[i]>volume_ma_slow[i]:
                volume_diff.append(1)
            elif volume_ma_fast[i]<volume_ma_slow[i]:
                volume_diff.append(3)
            else:
                volume_diff.append(5)
        df["volume1_5_10"] = volume_diff

    if 'volume2_5_10' in params:
        volume_ma_fast = talib.MA(volume, timeperiod=5, matype=0)
        volume_ma_slow = talib.MA(volume, timeperiod=10, matype=0)
        volume_ma_fast = np.array([(volume_ma_fast[i] if volume_ma_fast[i]>0 else 1) for i in range(0,len(volume_ma_fast))]) # remove zeroes
        volume_ma_slow = np.array([(volume_ma_slow[i] if volume_ma_slow[i] > 0 else 1) for i in range(0, len(volume_ma_slow))])  # remove zeroes

        volume_var = []
        for i in range(0,len(volume_ma_fast)):
            volume_var.append(volume_ma_fast[i]/volume_ma_fast[max(0,i-1)])
        df["volume2_5_10"] = volume_ma_fast - volume_ma_slow
        # df["volume2var_5_10"] = volume_var

    if 'volume2var_5' in params:
        volume_ma_fast = talib.MA(volume, timeperiod=5, matype=0)
        volume_ma_fast = np.array([(volume_ma_fast[i] if volume_ma_fast[i]>0 else 1) for i in range(0,len(volume_ma_fast))]) # remove zeroes
        volume_var = []
        for i in range(0,len(volume_ma_fast)):
            volume_var.append(volume_ma_fast[i]/volume_ma_fast[max(0,i-1)])
        df["volume2var_5"] = volume_var


    if 'pattern' in params:
        open = open_
        pattern = talib.CDL3LINESTRIKE(open, high, low, close)  # daily range
        df["pattern"] = pattern
        pattern2 = talib.CDLDOJISTAR(open, high, low, close)  # daily range
        df["pattern2"] = pattern2
        pattern3 = talib.CDLMORNINGSTAR(open, high, low, close, penetration=0)  # daily range
        df["pattern3"] = pattern3
        pattern4 = talib.CDLUNIQUE3RIVER(open, high, low, close)  # daily range
        df["pattern4"] = pattern4
        pattern5 = talib.CDL3WHITESOLDIERS(open, high, low, close)  # daily range
        df["pattern5"] = pattern5



    # *************************** ADD NEW FEATURES HERE BELLOW ***************************






    # df.to_csv(".\features_model.csv",sep="\t") # export features to a csv

    return df

