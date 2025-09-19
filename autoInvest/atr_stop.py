def atr_stop_exit(prices, atr, entry_idx, rr_ratio=1.2):
    """
    ATR 기반 손절/익절 청산 시점 산출 함수

    :param prices: 가격 리스트
    :param atr: ATR 지표 리스트
    :param entry_idx: 진입 인덱스
    :param rr_ratio: 이익 실현 비율 (기존 1.5에서 1.2로 완화하여 더 빠른 익절 가능)
    :return: 청산 인덱스, 청산 유형 ('stop' 또는 'take'), 조건 미충족 시 None, None
    """
    entry_price = prices[entry_idx]

    # 손절 기준은 진입가에서 ATR만큼 하락 시점
    stop_loss = entry_price - atr[entry_idx]

    # 익절 기준은 진입가에서 ATR의 rr_ratio 배만큼 상승 시점
    take_profit = entry_price + rr_ratio * atr[entry_idx]

    for i in range(entry_idx + 1, len(prices)):
        # 트레일링 스탑 계산: 이전 손절선과 현재 가격-ATR 중 큰 값을 택함 (익절에 가깝게 조정)
        trailing_stop = max(stop_loss, prices[i] - atr[i])

        # 가격이 트레일링 스탑 이하 도달 시 손절 처리
        if prices[i] <= trailing_stop:
            return i, 'stop'

        # 가격이 익절 목표에 도달 시 익절 처리
        if prices[i] >= take_profit:
            return i, 'take'

    # 종료 조건 미충족 시 None 반환
    return None, None
