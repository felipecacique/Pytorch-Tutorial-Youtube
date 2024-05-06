
import numpy as np

def transform_rates(time, open, high, low, close, timeframe, n):
    """This function receives rates (datetime, open, high, low, close) of a small timeframe and generates daily (or monthly) rates,
    of the same length and align with the rates array

    :param time: array with datetime
    :param open: array with open prices
    :param high: array with high prices
    :param low: array with low prices
    :param close: array with close prices
    :return opens: array of arrays with open prices in daily timeframe
    :return highs: array of arrays with high prices in daily timeframe
    :return lows: array of arrays with lows prices in daily timeframe
    :return closes: array of arrays with close prices in daily timeframe
    """
    opens = [[0 for i in range(0,len(time))] for j in range(0,n+1)]
    highs = [[0 for i in range(0, len(time))] for j in range(0,n+1)]
    lows = [[0 for i in range(0, len(time))] for j in range(0,n+1)]
    closes = [[0 for i in range(0, len(time))] for j in range(0,n+1)]

    for i, t in enumerate(time):
        if      (i ==0) or \
                (timeframe == "day" and time[i].day != time[max(0,i-1)].day) or \
                (timeframe == "month" and time[i].month != time[max(0,i-1)].month) or \
                (timeframe == "week" and time[i].weekday() < time[max(0,i -1)].weekday()) or \
                    (timeframe == "m5" and (time[i].hour * 60 + time[i].minute) % 5 < (time[max(0, i - 1)].hour * 60 + time[max(0, i - 1)].minute) % 5) or \
                    (timeframe == "m10" and (time[i].hour * 60 + time[i].minute) % 10 < (time[max(0, i - 1)].hour * 60 + time[max(0, i - 1)].minute) % 10) or \
                    (timeframe == "m15" and (time[i].hour * 60 + time[i].minute) % 15 < (time[max(0, i - 1)].hour * 60 + time[max(0, i - 1)].minute) % 15) or \
                    (timeframe == "m20" and (time[i].hour * 60 + time[i].minute) % 20 < (time[max(0, i - 1)].hour * 60 + time[max(0, i - 1)].minute) % 20) or \
                    (timeframe == "m30" and (time[i].hour * 60 + time[i].minute) % 30 < (time[max(0, i - 1)].hour * 60 + time[max(0, i - 1)].minute) % 30) or \
                    (timeframe == "h1" and (time[i].hour * 60 +time[i].minute) % 60 < (time[max(0,i -1)].hour * 60 + time[max(0,i -1)].minute)%60)  or \
                    (timeframe == "h2" and (time[i].hour * 60 + time[i].minute) % 120 < (time[max(0, i - 1)].hour * 60 + time[max(0, i - 1)].minute) % 120):    # we have a new day or month or candle of a specified timeframe
            # we have closed a last day - let's update the closes arrays

            # rolling the prices between days - close.n = close.n-1, ... close.2 = close.1, close.1 = close.0
            for j in list(reversed(range(1,n+1))):
                closes[j][i] = closes[j-1][i-1]
                opens[j][i] = opens[j-1][i-1]
                highs[j][i] = highs[j-1][i-1]
                lows[j][i] = lows[j-1][i-1]

            # update with the new day values
            closes[0][i] = close[i]
            opens[0][i] = open[i]
            highs[0][i] = high[i]
            lows[0][i] = low[i]

        else:
            closes[0][i] = close[i]
            opens[0][i] = opens[0][max(0,i-1)]
            highs[0][i] = max(highs[0][max(0,i-1)], high[i])
            lows[0][i] = min(lows[0][max(0,i-1)], low[i])

            # rolling the prices inside days - the next price is equal to the last
            for j in list(reversed(range(1,n+1))):
                closes[j][i] = closes[j][max(0,i-1)]
                opens[j][i] = opens[j][max(0,i-1)]
                highs[j][i] = highs[j][max(0,i-1)]
                lows[j][i] = lows[j][max(0,i-1)]

    return opens, highs, lows, closes


#
# def gap(open_0, close_1, mode=""):
#     """This function calculates the gap between days
#
#     :param open0: array with open price of the current day
#     :param close1: array with close price of the last day
#     :return: array with gaps
#     """
#     if mode == "percent": return [(open_0[i] - close_1[i])/close_1[i] for i in range(0, len(open_0))]
#     else: return [open_0[i] - close_1[i] for i in range(0,len(open_0))]
#
#
# def ibs(close, high_n, low_n):
#     """This function calculates the internal bar strength
#
#     :param close: array with close prices
#     :param highn: array with high prices of the last n days window
#     :param lown: array with low prices of the last n days window
#     :return: array with ibs
#     """
#     return [(close[i] - low_n[i])/(high_n - low_n) for i in range(0, len(close))]
#
#
# def cto(close, open_n, mode=""):
#     """This function calculates the close to open
#
#     :param close: array with close prices
#     :param open_n: array with open prices of the last n days window
#     :return: array with cto
#     """
#     if mode == "percent": return [(close[i] - open_n[i])/open_n[i] for i in range(0, len(close))]
#     else: return [close[i] - open_n[i] for i in range(0, len(close))]


def change(price1, price2, mode=""):
    """This function calculates the change between 2 prices

    :param price1: array with prices
    :param price2: array with prices
    :return: array with change
    """
    if mode == "percent": return [((price1[i] - price2[i])/price2[i] if price2[i] != 0 else 0) for i in range(0, len(price1))]
    if mode == "direction": return [(1 if (price1[i] - price2[i]) >= 0 else 0) for i in range(0, len(price1))]
    else: return [price1[i] - price2[i] for i in range(0, len(price1))]


def price_to_range(price1, price2, price3):
    """This function calculates the price to range

    :param price1: array with close prices
    :param price2: array with high prices
    :param price3: array with low prices
    :return: array with price to range
    """
    return [((price1[i] - price3[i])/(price2[i] - price3[i]) if (price2[i] - price3[i]) != 0 else 0) for i in range(0, len(price1))]


def relative_position(price1, price2, price3):
    """This function calculates the relative position

    :param price1: array with prices
    :param price2: array with prices
    :param price3: array with prices
    :return: array with price to range
    """
    range = price_to_range(price1, price2, price3)
    position = []
    for r in range:
        if r<0: position.append(0)  # position bellow
        elif r<0.5: position.append(1)  # position down
        elif r<=1.0: position.append(2)  # position up
        else: position.append(3)  # position above

    return position


def mean(X,exception=None):
    """This functions receives a list of lists and return a single list with the mean of each line

    :param X: list of lists
    :param exception: a value that will be discharged during the averaging
    :return: array with mean prices
    """
    means = []
    for i in range(0,len(X[0])):
        m = 0
        count = 0
        for j in range(0,len(X)):
            if (exception == None) or (exception != None and X[j][i] != exception):
                m += X[j][i]
                count += 1

        if count != 0:  m = m/count
        else: m = 0

        means.append(m)
    return means


def failure_high(high,open,close,mode=""):
    """This function calculates the failure high

    :param high: array with high prices
    :param open: array with open prices
    :param close: array with close prices
    :return: array with failure high
    """

    if mode == "percent": return [((high[i]-open[i])/open[i] if (close[i]<=open[i] and open[i]!=0) else -1) for i in range(0, len(close))]
    else:  return [(high[i]-open[i] if close[i]<=open[i] else -1) for i in range(0, len(close))]


def failure_low(low, open, close, mode=""):
    """This function calculates the failure high

    :param low: array with low prices
    :param open: array with open prices
    :param close: array with close prices
    :return: array with failure high
    """

    if mode == "percent":
        return [((open[i] - low[i]) / low[i] if (close[i]>=open[i] and open[i]!=0) else -1) for i in range(0, len(close))]
    else:
        return [(open[i] - low[i] if close[i] >= open[i] else -1) for i in range(0, len(close))]


def failure_size(open, high, low, mode=""):
    """This function calculates the failure size (get the minimum between open-high and open-low)

    :param low: array with low prices
    :param open: array with open prices
    :param close: array with close prices
    :return: array with failure size
    """

    if mode == "percent":
        return [(min((open[i] - low[i]) / open[i],(high[i] - open[i]) / open[i]) if open[i]!=0 else -1)  for i in range(0, len(open))]
    else:
        return [(min(open[i] - low[i], high[i] - open[i]) if open[i]!=0 else -1) for i in range(0, len(open))]


def daily_position(high1, low1, high2, low2):
    """This function gets the relative position between 2 days - (inside, outside, below, above)

    :param high1: list with daily high of the day 1
    :param low1: list with daily low of the day 1
    :param high2: list with daily high of the day 2
    :param low2: list with daily low of the day 2
    :return: array with the daily position
    """

    dayPosition= []
    for i in range(0, len(high1)):
        if high1[i]<=high2[i] and low1[i]<=low2[i]: dayPosition.append(0)      # day 1 is below day 2
        elif high1[i]<=high2[i] and low1[i]>=low2[i]: dayPosition.append(1)    # day 1 is inside day 2
        elif high1[i]>=high2[i] and low1[i]<=low2[i]: dayPosition.append(2)    # day 1 is outside day 2
        elif high1[i]>=high2[i] and low1[i]>=low2[i]: dayPosition.append(3)    # day 1 is above day 2
        else: dayPosition.append(4)
    return dayPosition


def label(indicator, intervals):
    """This function receive an indicator and discrete it according to the specified intervals (get real values and create labels).
    E.G., intervals=[0.3,0.5,0.7] imply in label 0 if indicator <= 0.3, label 1 if 0.3<indicator<=0.5, label 2 if 0.5<indicator<=0.7, and so on ...

    :param indicator: a list with values that divides the indicator and create labels
    :return: a list with labels
    """
    indicator = np.array(indicator)
    labels = [-1 for i in indicator]
    for i, value in enumerate(indicator):
        for j, q in enumerate(intervals):
            if value <= q:
                labels[i] = j
                break

            if j == len(intervals)-1:
                labels[i] = j + 1
    return labels


def month_quadrant(time, open_month, close_month):
    """This function the monthly quadrant

    :param time: array with datetime
    :param open_month: array with monthly open prices
    :param close_month: array with monthly close prices
    :return: array with labels
    """
    quadrant = []
    for i in range(0, len(time)):
        if time[i].day <= 10 and close_month[i] >= open_month[i]: quadrant.append(0)
        elif time[i].day <= 10 and close_month[i] < open_month[i]: quadrant.append(1)
        elif time[i].day <= 20 and close_month[i] >= open_month[i]: quadrant.append(2)
        elif time[i].day <= 20 and close_month[i] < open_month[i]: quadrant.append(3)
        elif time[i].day <= 31 and close_month[i] >= open_month[i]: quadrant.append(4)
        elif time[i].day <= 31 and close_month[i] < open_month[i]: quadrant.append(5)
        else: quadrant.append(6)

    return quadrant

def cmax(list):

    list_max = []
    for i, item in enumerate(list[0]):
        max_ = item
        for l in list:
            max_ = max(l[i],max_)
        list_max.append(max_)

    return list_max

def cmin(list):

    list_min = []
    for i, item in enumerate(list[0]):
        min_ = item
        for l in list:
            min_ = min(l[i],min_)
        list_min.append(min_)

    return list_min

def run(list_of_list, n):

    run_array = []
    for i in range(0,len(list_of_list[0])):
        r = 0
        for j,list in enumerate(list_of_list):
            r += list[i]*(n**j)
        run_array.append(r)

    return run_array

if __name__ == "__main__":
    from mydata import copy_rates_range, RatesMT5toRates
    import MetaTrader5 as mt5
    from datetime import datetime
    import numpy as np

    mt5.initialize(r"C:\Program Files\Metatrader 5  Terra\terminal64.exe")  # receive the path of the .exe of the metatrader 5 we want to use, and initialize the MetaTrader5 object

    ratesMT5 = copy_rates_range("WIN$N", mt5.TIMEFRAME_M30, datetime(2019, 11, 1), datetime(2021, 2, 15))  # getting rates from mt5 in raw format (list of lists)
    rates = RatesMT5toRates(ratesMT5)[:]  # converting raw rates into a list of dictionaries containing the keys: time, open, high, low, close, volume
    #
    # from mydata import CleanData
    # CleanData(rates)


    import pandas as pd
    df = pd.DataFrame()

    print("Calculating indicators!!")
    #--- separating some useful variables in order to use them for calculating some features
    time = np.array([rate["time"] for rate in rates])
    open_ = np.array([rate["open"] for rate in rates])
    high = np.array([rate["high"] for rate in rates])
    low = np.array([rate["low"] for rate in rates])
    close = np.array([rate["close"] for rate in rates])
    volume = np.array([float(rate["volume"]) for rate in rates])

    df["time"] = time
    df["open"] = open_
    df["high"] = high
    df["low"] = low
    df["close"] = close
    df["volume"] = volume

    #--- Getting daily, weekly and monthly timeframes
    open_day, high_day, low_day, close_day = transform_rates(time, open_, high, low, close, timeframe="day", n=30)
    open_week, high_week, low_week, close_week = transform_rates(time, open_, high, low, close, timeframe="week", n=5)
    open_month, high_month, low_month, close_month = transform_rates(time, open_, high, low, close, timeframe="month", n=1)
    open_h1, high_h1, low_h1, close_h1 = transform_rates(time, open_, high, low, close, timeframe="h1",n=2)

    df["open_h1_0"] = open_h1[0]
    df["high_h1_0"] = high_h1[0]
    df["low_h1_0"] = low_h1[0]
    df["close_h1_0"] = close_h1[0]

    df["open_h1_1"] = open_h1[1]
    df["high_h1_1"] = high_h1[1]
    df["low_h1_1"] = low_h1[1]
    df["close_h1_1"] = close_h1[1]

    df["open_day_0"] = open_day[0]
    df["high_day_0"] = high_day[0]
    df["low_day_0"] = low_day[0]
    df["close_day_0"] = close_day[0]

    df["open_day_1"] = open_day[1]
    df["high_day_1"] = high_day[1]
    df["low_day_1"] = low_day[1]
    df["close_day_1"] = close_day[1]

    df["open_day_2"] = open_day[2]
    df["open_day_3"] = open_day[3]
    df["open_week_0"] = open_week[0]
    df["open_week_1"] = open_week[1]
    df["open_week_2"] = open_week[2]
    df["open_month_0"] = open_month[0]
    df["high_month_0"] = high_month[0]
    df["low_month_0"] = low_month[0]
    df["close_month_0"] = close_month[0]
    df["open_month_1"] = open_month[1]
    df["high_month_1"] = high_month[1]
    df["low_month_1"] = low_month[1]
    df["close_month_1"] = close_month[1]


    #---
    # cto1 = change(close_day[1], open_day[1], "direction")   # direction (above or below 0) of close.1-open.1
    # cto2 = change(close_day[2], open_day[2], "direction")   # direction (above or below 0) of close.2-open.2
    # cto3 = change(close_day[3], open_day[3], "direction")   # direction (above or below 0) of close.3-open.3
    #
    cto1 = change(close_day[1], open_day[1], "percent")   # direction (above or below 0) of close.1-open.1
    cto2 = change(close_day[2], open_day[2], "percent")   # direction (above or below 0) of close.2-open.2
    cto3 = change(close_day[3], open_day[3], "percent")   # direction (above or below 0) of close.3-open.3

    df["cto1"] = cto1
    df["cto2"] = cto2
    df["cto3"] = cto3

    fopen0 = change(close, open_day[0], "percent")  # close from daily open (CTO)
    fhigh0 = change(close, high_day[0], "percent")  # close from daily high
    flow0 = change(close, low_day[0], "percent")    # close from daily low
    # fhigh4 = change(close, high_day[4], "percent")  # close from high.n
    # flow4 = change(close, low_day[4], "percent")    # close from low.n
    fhigh4 = change(close, cmax(high_day[0:5]), "percent")  # close from high.n
    flow4 = change(close, cmin(low_day[0:5]), "percent")    # close from low.n
    #--- IBS
    ibs0 = price_to_range(close, high_day[0], low_day[0])
    # ibs1 = price_to_range(close, high_day[1], low_day[1])
    # ibs4 = price_to_range(close, high_day[4], low_day[4])
    ibs1 = price_to_range(close, cmax(high_day[0:2]), cmin(low_day[0:2]))
    ibs4 = price_to_range(close, cmax(high_day[0:5]), cmin(low_day[0:5]))

    # c = cmax(high_day[0:3])
    # for i in range(0,len(high_day[0])):
    #     print(high_day[0][i], high_day[1][i], high_day[2][i], max(high_day[0][i],high_day[1][i],high_day[2][i]), c[i])
    #
    cmax4 = cmax(high_day[0:5])
    cmin4 = cmin(low_day[0:5])
    df["cmax4"] = cmax4
    df["cmin4"] = cmin4

    df["fopen0"] = fopen0
    df["fhigh0"] = fhigh0
    df["flow0"] = flow0
    df["fhigh4"] = fhigh4
    df["flow4"] = flow4

    df["ibs0"] = ibs0
    df["ibs1"] = ibs1
    df["ibs4"] = ibs4


    gap = change(open_day[0],close_day[1],"percent")

    df["gap"] = gap

    range0 = change(high_day[0], low_day[0], "percent")     # daily range
    range1 = change(high_day[1], low_day[1], "percent")  # daily range
    range_week0 = change(high_week[0],low_week[0],"percent")
    # range_week1 = change(high_week[1], low_week[1], "percent")
    range_week1 = change(cmax(high_week[0:2]), cmin(low_week[0:2]), "percent")
    range_month0 = change(high_month[0], low_month[0], "percent")

    average_range = mean([change(high_day[i], low_day[i], "percent") for i in range(1,10)])     # average "n" days range

    average_failure_high = mean([failure_high(high_day[i], open_day[i], close_day[i], "percent") for i in range(1,30)], exception=-1)
    average_failure_low = mean([failure_low(low_day[i], open_day[i], close_day[i], "percent") for i in range(1, 30)], exception=-1)

    df["range0"] = range0
    df["range1"] = range1
    df["range_week0"] = range_week0
    df["range_week1"] = range_week1
    df["range_month0"] = range_month0
    df["average_range"] = average_range
    df["average_failure_high"] = average_failure_high
    df["average_failure_low"] = average_failure_low

    #---  the daily position
    daily_position0 = daily_position(high_day[0], low_day[0], high_day[1], low_day[1])
    daily_position1 = daily_position(high_day[1], low_day[1], high_day[2], low_day[2])
    daily_position2 = daily_position(high_day[2], low_day[2], high_day[3], low_day[3])

    weekly_position0 = daily_position(high_week[0], low_week[0], high_week[1], low_week[1])


    relative_position0 = relative_position(close, high_day[0], low_day[0])

    df["daily_position0"] = daily_position0
    df["daily_position1"] = daily_position1
    df["daily_position2 "] = daily_position2
    df["weekly_position0"] = weekly_position0
    df["relative_position0"] = relative_position0

    hour = [time[i].hour for i, item in enumerate(time)]

    day = [time[i].day for i, item in enumerate(time)]

    day_label = label([time[i].day for i, item in enumerate(time)],[10,20])

    quadrant = month_quadrant(time, open_month[0], close_month[0])

    df["hour"] = hour
    df["day"] = day
    df["day_label"] = day_label
    df["quadrant"] = quadrant

    # [print(average_failure_high[i]) for i in range(0,len(average_range))]
    # # [print(time[i], open_day[0][i],high_day[0][i],low_day[0][i],close_day[0][i], open_day[1][i], open_day[2][i]) for i in range(0,len(open_day[0]))]


    cto1_label = label(change(close_day[1], open_day[1], "direction"),[0])  # direction (above or below 0) of close.1-open.1
    cto2_label = label(change(close_day[2], open_day[2], "direction"),[0])  # direction (above or below 0) of close.2-open.2
    cto3_label = label(change(close_day[3], open_day[3], "direction"),[0])  # direction (above or below 0) of close.3-open.3

    # RUN de cto
    run_cto = run([cto3_label,cto2_label,cto1_label],2)

    df["cto1_label"] = cto1_label
    df["cto2_label"] = cto2_label
    df["cto3_label"] = cto3_label

    df["run_cto"] = run_cto

    # RUN de daily position
    run_daily_position = run([daily_position2, daily_position1, daily_position0], 4)

    df["run_daily_position"] = run_daily_position

    df.to_csv(".\salib_features.csv",sep="\t")