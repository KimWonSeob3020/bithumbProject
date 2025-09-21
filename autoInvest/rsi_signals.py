import talib
import numpy as np

def rsi_signals(close_prices, period=14, low=40, high=60):
    """
    RSI 지표를 기반으로 매수(1), 매도(-1), 관망(0) 신호 생성
    :param close_prices: 종가 리스트
    :param period: RSI 계산 기간
    :param low: 매수 신호 발생 임계 RSI 값 (기존 50에서 40으로 낮춤)
    :param high: 매도 신호 발생 임계 RSI 값 (기존 55에서 60으로 높임)
    :return: 신호 배열 (1: 매수, -1: 매도, 0: 관망)
    """
    rsi = talib.RSI(np.array(close_prices), timeperiod=period)
    buy = (rsi < low) & (np.roll(rsi, 1) >= low)
    sell = (rsi > high) & (np.roll(rsi, 1) <= high)
    signals = np.where(buy, 1, np.where(sell, -1, 0))
    return signals
