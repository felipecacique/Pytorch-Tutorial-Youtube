#########################################################################################################################################
###--- This script contains the a set of many financial indicators that can be used as features for machine learning methods, etc. ---###
#########################################################################################################################################

import numpy as np
import talib

def GetclosesNdaywindow(rates, n, i, which = 'close'):
    """Necessary for Newton's feature - Gets the rates of n days ago"""
    closes = []  # this stores the closes prices in the range today to n days behind
    date = rates[i]["time"].date()  # stores the date of the current day
    day = 0  # count the number of days, starting from 0 that is the current day

    j = i
    while j >= 0:

        if date != rates[j]["time"].date():  # we have a new day
            day += 1

        if day > n:  # stop if we have already gotten all close prices within the range 0 to n (today to n days behind)
            break

        if which == 'close': closes.append(rates[j]["close"])  # collects the candles within the interval
        else: closes.append(rates[j]["open"])  # collects the candles within the interval

        date = rates[j]["time"].date()  # receive the date of the current candle
        j -= 1  # iterates backwards, starting from the current candle, and stops when n days has passed

    return list(reversed(closes))


def newton_features(rates):
    """Gets Newton1s features - max, min of the last 4 days, max, min of the last day, and today's change
    Rates must be a dictionary with: {time: datetime, open:value, high:value, low:value, close:value """

    fhigh4, flow4, fhigh0, flow0, fopen0 = [], [], [], [], []

    for i in range(0, len(rates)):

        # # get std of last n candles
        # rates_past = rates[max(0, i - 900):i + 1]
        # close = np.array([rate['close'] for rate in rates_past])
        # std = np.std(close)  # std in points
        # # std = np.mean([abs(rate['close'] - rate['open']) for rate in rates_past])  # std in points
        # std_pct = std / rates[i]['close'] + 0.0001  # std in percent of the price

        closes4daywindow = GetclosesNdaywindow(rates, 3, i) # gets a list of the close prices of 4 days ago
        closes0daywindow = GetclosesNdaywindow(rates, 0, i) # gets a list of the close prices of the current day
        opensToday = GetclosesNdaywindow(rates, 0, i, 'open')  # gets a list of the close prices of the current day

        high4 = max([rates[j]["close"] for j in range(max(0, i - 145), max(1, i-1))])
        low4 = min([rates[j]["close"] for j in range(max(0, i - 145), max(1, i-1))])
        fhigh4.append((rates[i]["close"] - high4) / high4)       # percentage change between the max of 4 days window and the current close price
        flow4.append((rates[i]["close"] - low4) / low4)          # percentage change between the min of 4 days window and the current close price

        # fhigh4.append((rates[i]["close"] - max(closes4daywindow)) / rates[i]["close"])  # percentage change between the max of 4 days and the current close price
        # flow4.append((rates[i]["close"] - min(closes4daywindow)) / rates[i]["close"])   # percentage change between the min of 4 days and the current close price
        fhigh0.append((rates[i]["close"] - max(closes0daywindow)) / max(closes0daywindow)) # percentage change between high and close of the current day
        flow0.append((rates[i]["close"] - min(closes0daywindow)) / min(closes0daywindow))  # percentage change between low and close of the current day
        fopen0.append((rates[i]["close"] - opensToday[0]) / opensToday[0])   # percentage change between open and close of the current day

    return fhigh4, flow4, fhigh0, flow0, fopen0


def newton_features2(rates):
    """Gets Newton1s features - max, min of the last 4 days, max, min of the last day, and today's variation
    Rates must be a dictionary with: {time: datetime, open:value, high:value, low:value, close:value """
    fhigh4, flow4, fhigh0, flow0, fopen0 = [], [], [], [], []
    for i in range(0, len(rates)):
        closes4daywindow = GetclosesNdaywindow(rates, 3, i) # gets a list of the close prices of 4 days ago
        closes0daywindow = GetclosesNdaywindow(rates, 0, i) # gets a list of the close prices of the current day
        opensToday = GetclosesNdaywindow(rates, 0, i, 'open')  # gets a list of the close prices of the current day

        high4 = max([rates[j]["close"] for j in range(max(0, i - 145), max(1, i-1))])
        low4 = min([rates[j]["close"] for j in range(max(0, i - 145), max(1, i-1))])
        fhigh4.append((rates[i]["close"] - high4) / opensToday[0])
        flow4.append((rates[i]["close"] - low4) / opensToday[0])

        # fhigh4.append((rates[i]["close"] - max(closes4daywindow)) / rates[i]["close"])  # percentual variation between the max of 4 days and the current close price
        # flow4.append((rates[i]["close"] - min(closes4daywindow)) / rates[i]["close"])
        fhigh0.append((max(closes0daywindow)-opensToday[0]) / opensToday[0])
        flow0.append((min(closes0daywindow)-opensToday[0]) / opensToday[0])
        fopen0.append((rates[i]["close"] - opensToday[0]) / opensToday[0])

    return fhigh4, flow4, fhigh0, flow0, fopen0


def breakout_ret(rates, n=32):
    """Gets the max and min candle variations of the last n candles, and calculate the percentage change of the current candle, compared to the max and min """
    breakoutMaxRet, breakoutMinRet = [], []
    for i in range(0, len(rates)):
        max_candles = np.max(np.array([rates[j]['close'] - rates[j]['open'] for j in range(max(0, i - n), max(1, i - 1))]))
        min_candles = np.min(np.array([rates[j]['close'] - rates[j]['open'] for j in range(max(0, i - n), max(1, i - 1))]))
        breakoutMaxRet.append((rates[-1]['close'] - rates[-1]['open']) / max(2 * max_candles, 0.00005))
        breakoutMinRet.append((rates[-1]['close'] - rates[-1]['open']) / min(2 * min_candles, -0.00005))
    return breakoutMaxRet, breakoutMinRet


def getGaps(rates):
    """This function gets the gap of between days"""
    gaps = [0 for i in range(0, len(rates))]
    for i in range(0, len(rates)):
        if rates[max(i-1,0)]['time'].day != rates[i]['time'].day:
            gaps[i] = (rates[max(i-1,0)]['close'] - rates[i]['open']) / rates[i]['open']
        else:
            gaps[i] = gaps[max(i-1,0)]
    return gaps


def getOHLCYesterday(rates):
    """This functions gets the open, high, low, close of the previous day"""
    opens = [0 for i in range(0, len(rates))]
    highs = [0 for i in range(0, len(rates))]
    lows = [0 for i in range(0, len(rates))]
    closes = [0 for i in range(0, len(rates))]
    open_, high, low, close = rates[0]['open'], rates[0]['high'], rates[0]['low'], rates[0]['close']
    open_yesterday, high_yesterday, low_yesterday, close_yesterday = open_,high, low, close
    for i in range(0, len(rates)):
        if rates[max(i-1,0)]['time'].day != rates[i]['time'].day or i==0:   # we have a new day (first candle of the day)
            open_yesterday, high_yesterday, low_yesterday, close_yesterday = open_, high, low, close
            open_, high, low, close = rates[i]['open'], rates[i]['high'], rates[i]['low'], rates[i]['close']

        open_, high, low, close = open_, max(high, rates[i]['high']), min(low, rates[i]['low']), rates[i]['close']
        opens[i], highs[i], lows[i], closes[i] = open_yesterday, high_yesterday, low_yesterday, close_yesterday

    return opens, highs, lows, closes


def getPivotPoint(highs, lows, closes):
    """This function receives the daily hlc and calculates the pivot points"""
    p,r1,r2,r3,s1,s2,s3 = [],[],[],[],[],[],[]
    for i in range(0, len(highs)):
        high, low, close = highs[i], lows[i], closes[i]
        pivot = (high + close + low) / 3

        p.append(pivot)
        r1.append((2 * pivot) - low)
        s1.append((2 * pivot) - high)
        r2.append(pivot + (r1[-1] - s1[-1]))
        r3.append(high + (2 * (pivot - low)))
        s2.append(pivot - (r1[-1] - s1[-1]))
        s3.append(low - (2 * (high - pivot)))
    return p,r1,r2,r3,s1,s2,s3


def getCTR(opens, highs, lows, closes):
    """This function receives the daily ohlc and calculates the close to range"""
    ctr = []
    for i in range(0, len(highs)):
        ctr.append((closes[i]-lows[i])/(highs[i]-lows[i]))
    return ctr


def getCTO(opens, highs, lows, closes):
    """This function receives the daily ohlc and calculates the open to range"""
    cto = []
    for i in range(0, len(highs)):
        cto.append((closes[i]-opens[i])/(opens[i]))
    return cto


def getOHLCYesterdayN(rates, n):
    """This functions gets the open, high, low, close of the previous n day (one day only)"""
    opensN = [0 for i in range(0, len(rates))]
    highsN = [0 for i in range(0, len(rates))]
    lowsN = [0 for i in range(0, len(rates))]
    closesN = [0 for i in range(0, len(rates))]

    if n == 0:  # if we want the OHLC of the current day so far
        for i in range(0, len(rates)):
            closes0daywindow = GetclosesNdaywindow(rates, 0, i) # gets a list of the close prices of the current day
            opensToday = GetclosesNdaywindow(rates, 0, i, 'open')  # gets a list of the close prices of the current day
            opensN[i], highsN[i], lowsN[i], closesN[i] = opensToday[0], max(closes0daywindow), min(closes0daywindow), closes0daywindow[-1]

        return opensN, highsN, lowsN, closesN

    open_, high, low, close = [rates[0]['open']], [rates[0]['high']], [rates[0]['low']], [rates[0]['close']]
    open_n, high_n, low_n, close_n = open_[-1],high[-1], low[-1], close[-1]
    for i in range(0, len(rates)):
        if rates[max(i-1,0)]['time'].day != rates[i]['time'].day or i==0:   # we have a new day (first candle of the day)
            open_n, high_n, low_n, close_n = open_[max(0,len(open_)-n)], high[max(0,len(high)-n)], low[max(0,len(low)-n)], close[max(0,len(close)-n)]
            open_.append(rates[i]['open'])
            high.append(rates[i]['high'])
            low.append(rates[i]['low'])
            close.append(rates[i]['close'])

        open_[-1], high[-1], low[-1], close[-1] = open_[-1], max(high[-1], rates[i]['high']), min(low[-1], rates[i]['low']), rates[i]['close']
        opensN[i], highsN[i], lowsN[i], closesN[i] = open_n, high_n, low_n, close_n

    return opensN, highsN, lowsN, closesN


def getCloseToRange(close,high,low):
    """This function receives the daily hlc and calculates the close to range - where in % the price is between the high and low (we can use open prices as well)"""
    ctr = []
    for i in range(0, len(close)):
        if high[i] - low[i] != 0:
            ctr.append((close[i] - low[i]) / (high[i] - low[i]))
        else:
            ctr.append(0)
    return ctr


def getCandleId(rates):
    """This function gets the candle id - which candle is this? the first of the day, the second ... """
    candle_id = []
    id = 0
    for i, rate in enumerate(rates):
        if rates[i]['time'].day != rates[max(0,i-1)]['time'].day:
            id = 0
        candle_id.append(id)
        id += 1
    return candle_id


def movingAverageLabel(close, period_fast=9, period_slow=40):
    """Create labels using 2 MAs"""
    ma_fast = talib.MA(close, timeperiod=period_fast, matype=1)
    ma_slow = talib.MA(close, timeperiod=period_slow, matype=1)
    ma_label = []
    for i in range(0, len(ma_fast)):
        if ma_fast[i] <= ma_slow[i] and ma_fast[max(0, i - 1)] >= ma_slow[max(0, i - 1)]:   # ma fast cross below the ma slow
            ma_label.append(0)
        elif ma_fast[i] >= ma_slow[i] and ma_fast[max(0, i - 1)] <= ma_slow[max(0, i - 1)]:  # ma fast cross above the ma slow
            ma_label.append(1)
        elif ma_fast[i] >= ma_slow[i]:  # ma fast is above the ma slow
            ma_label.append(2)
        elif ma_fast[i] <= ma_slow[i]:  # ma fast is below the ma slow
            ma_label.append(3)
        else:
            ma_label.append(4)
    return ma_label, ma_fast, ma_slow


def rsiLabel(close,period=14):
    """Discretize the rsi levels"""
    rsi = talib.RSI(close, timeperiod=period)
    rsi_label = []
    for i in range(0, len(rsi)):
        if rsi[i] <= 20 and rsi[max(0, i - 1)] >= 20:   # rsi cross below the level 20
            rsi_label.append(0)
        elif rsi[i] >= 80 and rsi[max(0, i - 1)] <= 80: # rsi cross above the level 80
            rsi_label.append(1)
        elif rsi[i] <= 20:
            rsi_label.append(2)
        elif rsi[i] <= 30:
            rsi_label.append(3)
        elif rsi[i] <= 50:
            rsi_label.append(4)
        elif rsi[i] <= 70:
            rsi_label.append(5)
        elif rsi[i] <= 80:
            rsi_label.append(6)
        elif rsi[i] <= 100:
            rsi_label.append(7)
        else:
            rsi_label.append(8)
    return rsi_label, rsi


def macdSignalLabel(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """Create labels using the macd signal"""
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod, slowperiod, signalperiod)
    macd_label = []
    for i in range(0, len(macd)):
        if macdsignal[i] <= macd[i] and macdsignal[max(0, i - 1)] >= macd[max(0, i - 1)]:   # macd signal crosses below the macd
            macd_label.append(0)
        elif macdsignal[i] >= macd[i] and macdsignal[max(0, i - 1)] <= macd[max(0, i - 1)]:  # macd signal above below the macd
            macd_label.append(1)
        elif macdsignal[i] >= macd[i]:  # macd signal above macd
            macd_label.append(2)
        elif macdsignal[i] <= macd[i]:  # macd signal below macd
            macd_label.append(3)
        else:
            macd_label.append(4)
    return macd_label, macd , macdsignal, macdhist


def macdHistLabel(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """Create labels using the macd histogram and signal"""
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod, slowperiod, signalperiod)
    macd_label = []
    for i in range(0, len(macd)):
        if macdhist[i] <= 0 and macdsignal[max(0, i - 1)] >= 0:
            macd_label.append(0)
        elif macdhist[i] <= 0 and macdsignal[max(0, i - 1)] <= 0:
            macd_label.append(1)
        elif macdhist[i] >= 0 and macdsignal[max(0, i - 1)] <= 0:
            macd_label.append(2)
        elif macdhist[i] >= 0 and macdsignal[max(0, i - 1)] >= 0:
            macd_label.append(3)
        else:
            macd_label.append(4)
    return macd_label, macd , macdsignal, macdhist


def stochLabel(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    "Discretizing the stochastic levels"
    slowk, slowd = talib.STOCH(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)
    sto_label = []
    for i in range(0, len(slowk)):
        if slowk[i] <= 20:
            sto_label.append(0)
        elif slowk[i] <= 50:
            sto_label.append(1)
        elif slowk[i] <= 80:
            sto_label.append(2)
        elif slowk[i] <= 100:
            sto_label.append(3)
        else:
            sto_label.append(4)
    return sto_label, slowk, slowd


def stochLabel2(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    "Creating labels based on stochastic k and d"
    slowk, slowd = talib.STOCH(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)
    sto_label = []
    for i in range(0, len(slowk)):
        if slowk[i] <= slowd[i] and slowk[max(0, i - 1)] >= slowd[max(0, i - 1)]:
            sto_label.append(0)
        elif slowk[i] >= slowd[i] and slowk[max(0, i - 1)] <= slowd[max(0, i - 1)]:
            sto_label.append(1)
        elif slowk[i] >= slowd[i] and slowk[max(0, i - 1)] >= slowd[max(0, i - 1)]:
            sto_label.append(2)
        elif slowk[i] <= slowd[i] and slowk[max(0, i - 1)] <= slowd[max(0, i - 1)]:
            sto_label.append(3)
        else:
            sto_label.append(4)
    return sto_label, slowk, slowd


def bbandsLabel(close, timeperiod=14, nbdevup=2, nbdevdn=2, matype=0):
    "Creating labels based on the bollinger bands"
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod, nbdevup, nbdevdn, matype)
    bb_label = []
    for i in range(0, len(upperband)):
        if close[i] <= lowerband[i] and close[max(0, i - 1)] >= lowerband[max(0, i - 1)]:   # price crossed below the lower bollinger bands
            bb_label.append(0)
        elif close[i] >= upperband[i] and close[max(0, i - 1)] <= upperband[max(0, i - 1)]: # price crossed above the upper bollinger bands
            bb_label.append(1)
        elif close[i] <= lowerband[i]:  # price below lower bb
            bb_label.append(2)
        elif close[i] <= middleband[i]: # price below middle bb
            bb_label.append(3)
        elif close[i] <= upperband[i]: # price below upper bb
            bb_label.append(4)
        elif close[i] > upperband[i]: # price above upper bb
            bb_label.append(5)
        else:
            bb_label.append(6)
    return bb_label, upperband, middleband, lowerband


def getPolarityDay(rates,day):
    """Get the pertentage change of a specified day, the close to range and the polarity(if close below open, or close above open).
    If day_interval is 1, it means we are getting the variation of yesterday"""
    opens1, highs1, lows1, closes1 = getOHLCYesterdayN(rates,day)
    d1_polarity = []
    d1_close_to_range = []
    d1_var = []
    for i in range(0, len(rates)):
        d1_polarity.append((1 if (closes1[i] - opens1[i]) / (opens1[i]) > 0 else 0))
        d1_var.append( (closes1[i] - opens1[i]) / (opens1[i]))

        if highs1[i] - lows1[i] != 0:
            d1_close_to_range.append((closes1[i] - lows1[i]) / (highs1[i] - lows1[i]))
        else:
            d1_close_to_range.append(0)

    return d1_polarity, d1_close_to_range, d1_var


def getPolarityDayInterval(rates, day):
    """Get the percentage change of an interval of days, the close to range and the polarity(close below open, or close above open).
    I day_interval is 21, it means we are getting the variation of the last 21 days (last month*)"""
    opens1, highs1, lows1, closes1 = getOHLCYesterdayN(rates, day)      # gets the ohlc of the n-th days ago
    d1_polarity = []            # polarity(close below open, or close above open)
    d1_close_to_range = []      # close to range
    d1_var = []                 # percentage change of an interval of days
    for i in range(0, len(rates)):
        opensToday = GetclosesNdaywindow(rates, 0, i, 'open')   # gets a list of the open prices of the current day
        closesInterval = GetclosesNdaywindow(rates, day, i)     # gets a list of the close prices of n days window
        interval_low = min(closesInterval)                      # gets the low price of the n days window
        interval_high = max(closesInterval)                     # gets the high price of the n days window

        d1_polarity.append((1 if (rates[i]['close'] - opens1[i]) / (opens1[i]) > 0 else 0))
        # d1_polarity.append((1 if (opensToday[0] - opens1[i]) / (opens1[i]) > 0 else 0))
        # d1_var.append((opensToday[0] - opens1[i]) / (opens1[i]))
        d1_var.append((rates[i]['close'] - opens1[i]) / (opens1[i]))
        # d1_polarity.append((1 if (opensToday[0] - opens1[i]) / (opens1[i]) > 0 else 0))
        # d1_var.append((opensToday[0] - opens1[i]) / (opens1[i]))

        if interval_high - interval_low != 0:
            # d1_close_to_range.append((opensToday[0] - interval_low) / (interval_high - interval_low))
            d1_close_to_range.append((rates[i]['close'] - interval_low) / (interval_high - interval_low))
            # d1_close_to_range.append((opensToday[0] - interval_low) / (interval_high - interval_low))
        else:
            d1_close_to_range.append(0)

    return d1_polarity, d1_close_to_range, d1_var


def pivotLabel(rates):
    """Creating labels using the pivot points"""
    close = np.array([rate["close"] for rate in rates])
    opens, highs, lows, closes = getOHLCYesterday(rates)
    p, r1, r2, r3, s1, s2, s3 = getPivotPoint(highs, lows, closes)

    pivot = []
    for i in range(0, len(p)):
        if close[i] <= p[i] and close[max(0, i - 1)] >= p[max(0, i - 1)]:
            pivot.append(0)
        elif close[i] >= p[i] and close[max(0, i - 1)] <= p[max(0, i - 1)]:
            pivot.append(1)

        elif close[i] <= s1[i] and close[max(0, i - 1)] >= s1[max(0, i - 1)]:
            pivot.append(2)
        elif close[i] >= s1[i] and close[max(0, i - 1)] <= s1[max(0, i - 1)]:
            pivot.append(3)

        elif close[i] <= s2[i] and close[max(0, i - 1)] >= s2[max(0, i - 1)]:
            pivot.append(4)
        elif close[i] >= s2[i] and close[max(0, i - 1)] <= s2[max(0, i - 1)]:
            pivot.append(5)

        elif close[i] <= s3[i] and close[max(0, i - 1)] >= s3[max(0, i - 1)]:
            pivot.append(6)
        elif close[i] >= s3[i] and close[max(0, i - 1)] <= s3[max(0, i - 1)]:
            pivot.append(7)

        elif close[i] <= r1[i] and close[max(0, i - 1)] >= r1[max(0, i - 1)]:
            pivot.append(8)
        elif close[i] >= r1[i] and close[max(0, i - 1)] <= r1[max(0, i - 1)]:
            pivot.append(9)

        elif close[i] <= r2[i] and close[max(0, i - 1)] >= r2[max(0, i - 1)]:
            pivot.append(10)
        elif close[i] >= r2[i] and close[max(0, i - 1)] <= r2[max(0, i - 1)]:
            pivot.append(11)

        elif close[i] <= r3[i] and close[max(0, i - 1)] >= r3[max(0, i - 1)]:
            pivot.append(12)
        elif close[i] >= r3[i] and close[max(0, i - 1)] <= r3[max(0, i - 1)]:
            pivot.append(13)



        elif close[i] <= s3[i]:
            pivot.append(14)
        elif close[i] >= s3[i] and close[max(0, i - 1)] <= s2[max(0, i - 1)]:
            pivot.append(15)

        elif close[i] >= s2[i] and close[max(0, i - 1)] <= s1[max(0, i - 1)]:
            pivot.append(16)
        elif close[i] >= s1[i] and close[max(0, i - 1)] <= p[max(0, i - 1)]:
            pivot.append(17)
        elif close[i] >= p[i] and close[max(0, i - 1)] <= r1[max(0, i - 1)]:
            pivot.append(18)
        elif close[i] >= r1[i] and close[max(0, i - 1)] <= r2[max(0, i - 1)]:
            pivot.append(19)
        elif close[i] >= r2[i] and close[max(0, i - 1)] <= r3[max(0, i - 1)]:
            pivot.append(20)
        elif close[i] >= r3[i] :
            pivot.append(21)

        else:
            pivot.append(22)
    return pivot


def getDayPosition(rates, highs1, highs2, lows1, lows2):
    """This function gets the relative position between 2 days - (inside, below, above)"""
    dayPosition= []
    for i in range(0, len(rates)):
        if highs1[i]<highs2[i] and lows1[i]<lows2[i]: dayPosition.append(0)      # day 1 is below day 2
        elif highs1[i]<highs2[i] and lows1[i]>lows2[i]: dayPosition.append(1)    # day 1 is inside day 2
        elif highs1[i]>highs2[i] and lows1[i]<lows2[i]: dayPosition.append(2)    # day 2 is inside day 1
        elif highs1[i]>highs2[i] and lows1[i]>lows2[i]: dayPosition.append(3)    # day 1 is above day 2
        else: dayPosition.append(4)
    return dayPosition


def closeToRangeLabel(prices, highs, lows, label_mode=0):
    """This function receives the daily hlc and create labels using close to range"""
    ctr = getCloseToRange(prices, highs, lows)

    ctr_label = []
    if label_mode == 0:
        for i in range(0, len(ctr)):
            if ctr[i] <= 0.50: ctr_label.append(0)
            else: ctr_label.append(1)

    elif label_mode == 1:
        for i in range(0, len(ctr)):
            if ctr[i] <= 0.33: ctr_label.append(0)
            elif ctr[i] <= 0.66: ctr_label.append(1)
            else: ctr_label.append(2)

    elif label_mode == 2:
        for i in range(0, len(ctr)):
            if ctr[i] <= 0.25: ctr_label.append(0)
            elif ctr[i] <= 0.50: ctr_label.append(1)
            elif ctr[i] <= 0.75: ctr_label.append(2)
            else: ctr_label.append(3)

    elif label_mode == 3:
        for i in range(0, len(ctr)):
            if ctr[i] <= 0.20: ctr_label.append(0)
            elif ctr[i] <= 0.40: ctr_label.append(1)
            elif ctr[i] <= 0.60: ctr_label.append(2)
            elif ctr[i] <= 0.80: ctr_label.append(3)
            else: ctr_label.append(4)

    else:
        for i in range(0, len(ctr)):
            if ctr[i] <= 0.10: ctr_label.append(0)
            elif ctr[i] <= 0.20: ctr_label.append(1)
            elif ctr[i] <= 0.30: ctr_label.append(2)
            elif ctr[i] <= 0.40: ctr_label.append(3)
            elif ctr[i] <= 0.50: ctr_label.append(4)
            elif ctr[i] <= 0.60: ctr_label.append(5)
            elif ctr[i] <= 0.70: ctr_label.append(6)
            elif ctr[i] <= 0.80: ctr_label.append(7)
            elif ctr[i] <= 0.90: ctr_label.append(8)
            else: ctr_label.append(9)

    return ctr_label


def indicatorToLabel(indicator, intervals):
    """This function receive an indicator and discretize it according to the specified intervals"""
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


def gapLabel(rates, levels_array = (-0.010, -0.005, 0.000, 0.005, 0.010)):
    """Create labels using close to range"""
    gaps = getGaps(rates)
    return indicatorToLabel(gaps, levels_array)


def getGabarito(opens, highs, lows, closes, period=30, quantile=.90):
    """Gets a distribution of percentage change of 'period' candles. Calculates the 'quantile'th higher return of the distribution.
    A quantile=.10 in a distribution of 100 returns means that we are getting the 10th highest return. A quantile=0.99 means that
    we are getting the worse return (the day that fall the most)"""
    gabarito=[]
    std_open_low = []
    std_open_high = []
    for i in range(0, len(opens)):
        distribution = [(closes[j] - opens[j]) / opens[j] for j in range(max(0, i - period), max(1, i))]
        gabarito.append(list(sorted(distribution))[min(int((1-quantile)*len(distribution)), len(distribution)-1)])
        std_open_low.append(np.std([(lows[j] - opens[j])/ opens[j]for j in range(max(0, i - period), max(1, i))]))
        std_open_high.append(np.std([(highs[j] - opens[j])/ opens[j]for j in range(max(0, i - period), max(1, i))]))
    return gabarito, std_open_low, std_open_high

def midLabel(prices, highs, lows):
    """A estrutura de preço MID.1 identifica a posição relativa do preço atual (de entrada no trade) em relação ao ponto
        médio, máxima e mínima do dia anterior"""
    ctr = getCloseToRange(prices, highs, lows)

    ctr_label = []
    for i in range(0, len(ctr)):
        if ctr[i] <= 0.0: ctr_label.append(0)
        elif ctr[i] <= 0.50: ctr_label.append(1)
        elif ctr[i] <= 1.0: ctr_label.append(1)
        else: ctr_label.append(3)
    return ctr_label, ctr