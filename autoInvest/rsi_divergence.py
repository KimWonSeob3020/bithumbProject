def rsi_divergence(prices, rsi, window=7):
    """
    가격과 RSI 지표 간의 다이버전스 감지

    강세 다이버전스: 가격은 낮은 저점 형성, RSI는 더 높은 저점 형성
    약세 다이버전스: 가격은 높은 고점 형성, RSI는 더 낮은 고점 형성

    :param prices: 가격 리스트
    :param rsi: RSI 지표 리스트
    :param window: 비교 기준 거리 (기존 10 -> 7로 축소하여 민감도 증대)
    :return: 강세 다이버전스 인덱스 리스트, 약세 다이버전스 인덱스 리스트
    """
    bullish, bearish = [], []
    for i in range(window, len(prices)-window):
        # 강세 다이버전스 조건: 가격이 중앙점에서 낮은 저점이나 RSI는 더 높은 저점 형성
        if prices[i] < prices[i-window] and prices[i] < prices[i+window]:
            if rsi[i] > rsi[i-window] and rsi[i] > rsi[i+window]:
                bullish.append(i)

        # 약세 다이버전스 조건: 가격은 높은 고점, RSI는 더 낮은 고점 형성
        if prices[i] > prices[i-window] and prices[i] > prices[i+window]:
            if rsi[i] < rsi[i-window] and rsi[i] < rsi[i+window]:
                bearish.append(i)
    return bullish, bearish