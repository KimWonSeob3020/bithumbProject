import numpy as np
from scipy.signal import savgol_filter

def find_support_resistance(close_prices, window=20):
    """
    가격 차트를 Savitzky-Golay 필터로 부드럽게 한 뒤,
    이동평균의 기울기 전환점을 찾아 지지선과 저항선 위치 반환

    :param close_prices: 종가 가격 리스트
    :param window: smoothing 및 기울기 판단용 윈도우 크기 (default 20)
                   이 값을 조절하여 좀 더 민감하거나 완화된 전환점 탐색 가능
    :return: 지지선, 저항선 위치 튜플 리스트 (인덱스, 가격)
    """

    # 가격 데이터를 Savitzky-Golay 필터로 부드럽게 처리하여 노이즈 감소
    smoothed = savgol_filter(np.array(close_prices), window, 3)

    # 부드러운 가격의 차분 계산 (기울기)
    diffs = np.diff(smoothed)

    support, resistance = [], []
    for i in range(window, len(diffs)-window):
        # 이전 window 구간이 모두 하락세이고 이후 window 구간이 모두 상승세면 지지선 전환점 판정
        if np.all(diffs[i-window:i] <= 0) and np.all(diffs[i:i+window] >= 0):
            support.append((i, close_prices[i]))

        # 이전 window 구간이 모두 상승세이고 이후 window 구간이 모두 하락세면 저항선 전환점 판정
        if np.all(diffs[i-window:i] >= 0) and np.all(diffs[i:i+window] <= 0):
            resistance.append((i, close_prices[i]))

    return support, resistance
