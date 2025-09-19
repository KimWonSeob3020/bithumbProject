import talib
import numpy as np

def rsi_signals(close_prices, period=14, low=50, high=55):
    """
    RSI 지표를 기반으로 매수(1), 매도(-1), 관망(0) 신호 생성

    :param close_prices: 종가 리스트
    :param period: RSI 계산 기간
    :param low: 매수 신호 발생 임계 RSI 값 (기존 40 -> 50으로 완화)
    :param high: 매도 신호 발생 임계 RSI 값 (기존 60 -> 55로 완화)
    :return: 신호 배열 (1: 매수, -1: 매도, 0: 관망)
    """

    rsi = talib.RSI(np.array(close_prices), timeperiod=period)
    # RSI가 low 아래로 내려가면 매수 신호 발생 (이전 값은 above low)
    buy = (rsi < low) & (np.roll(rsi, 1) >= low)
    # RSI가 high 위로 올라가면 매도 신호 발생 (이전 값은 below high)
    sell = (rsi > high) & (np.roll(rsi, 1) <= high)

    signals = np.where(buy, 1, np.where(sell, -1, 0))
    return signals