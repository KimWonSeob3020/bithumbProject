import numpy as np
from scipy.signal import savgol_filter

def find_support_resistance(close_prices, window=20, tolerance=20):
    """
    지지선과 저항선을 찾아 반환
    :param close_prices: 종가 리스트
    :param window: 스무딩 윈도우
    :param tolerance: 지지/저항 근접 허용 범위 (기존 ±10에서 ±20으로 완화)
    :return: support, resistance 리스트 (인덱스 및 가격)
    """
    smoothed = savgol_filter(np.array(close_prices), window, 3)
    diffs = np.diff(smoothed)

    support = []
    resistance = []
    for i in range(window, len(diffs) - window):
        if np.all(diffs[i - window:i] > 0) and np.all(diffs[i:i + window] < 0):
            resistance.append((i, close_prices[i]))
        elif np.all(diffs[i - window:i] < 0) and np.all(diffs[i:i + window] > 0):
            support.append((i, close_prices[i]))

    # tolerance 범위 활용은 main_trading.py에서 처리

    return support, resistance
