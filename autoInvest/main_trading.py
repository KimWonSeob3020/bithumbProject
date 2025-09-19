import time
from datetime import datetime
import talib
import numpy as np

from rsi_signals import rsi_signals
from support_resistance import find_support_resistance
from rsi_divergence import rsi_divergence
from atr_stop import atr_stop_exit
from position_manager import save_positions, load_positions
import python_bithumb_api

def main(ticker, prices, high, low, close):
    """
    자동매매 메인 함수
    :param ticker: 코인 심볼 정보
    :param prices: 시세 가격 리스트
    :param high: 고가 리스트
    :param low: 저가 리스트
    :param close: 종가 리스트
    :return: 매수/매도 포지션 리스트
    """

    # numpy array 형태로 변환하여 talib 함수 계산 정확도 향상
    prices = np.array(prices, dtype=np.float64)
    high = np.array(high, dtype=np.float64)
    low = np.array(low, dtype=np.float64)
    close = np.array(close, dtype=np.float64)

    # RSI 기반 매수/매도 신호 계산
    signals = rsi_signals(prices)

    # RSI 값 계산 (다이버전스 탐색용)
    rsi = talib.RSI(prices, timeperiod=14)
    # ATR(평균 진폭) 계산 - 손절/익절 청산 조건에서 활용
    atr = talib.ATR(high, low, close, timeperiod=14)

    # 스윙 지지/저항 위치 탐색 (전환점)
    support, resistance = find_support_resistance(prices)

    # RSI 다이버전스 위치 탐색 (강세/약세 구분)
    bullish_diverge, bearish_diverge = rsi_divergence(prices, rsi)

    positions = []
    for i in range(len(signals)):
        signal = signals[i]

        # === 지지/저항 근처 여부 판단 ===
        # 기존 ±10 범위 대신 ±20 범위로 확장하여 더 넓은 범위 인정
        near_support = any(abs(s[0] - i) <= 20 for s in support)
        near_resistance = any(abs(r[0] - i) <= 20 for r in resistance)

        # === 매수 조건 ===
        # RSI 매수신호 + 강세 다이버전스 + 지지선 근처
        if signal == 1 and i in bullish_diverge and near_support:
            # ATR 손절익절 함수에서 이익 실현 비율(rr_ratio) 1.2로 설정하여 빠른 익절 가능
            exit_idx, exit_type = atr_stop_exit(prices, atr, i, rr_ratio=1.2)
            positions.append({'ticker': ticker, 'entry': i, 'exit': exit_idx, 'type': 'buy', 'exit_type': exit_type})

        # === 매도 조건 ===
        # RSI 매도신호 + 약세 다이버전스 + 저항선 근처
        elif signal == -1 and i in bearish_diverge and near_resistance:
            exit_idx, exit_type = atr_stop_exit(prices, atr, i, rr_ratio=1.2)
            positions.append({'ticker': ticker, 'entry': i, 'exit': exit_idx, 'type': 'sell', 'exit_type': exit_type})

    # 포지션 리스트를 JSON 파일로 저장
    save_positions(positions, 'positions.json')
    return positions


if __name__ == "__main__":

    bithumb_api = python_bithumb_api.Bithumb_api()

    select_coin_list = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-BCH', 'KRW-AVAX', 'KRW-DOGE', 'KRW-TRX', 'KRW-ADA', 'KRW-SOL']
    coin_list = bithumb_api.get_market_all()

    while True:
        for coin in select_coin_list:
            try:
                ##ticker = coin['market']
                ticker = coin

                # OHLCV 데이터 로드 (5분봉 원하면 minute5로 변경)
                df = bithumb_api.get_ohlcv(ticker, "minute1")
                close_prices = df['close'].values.astype(np.float64)
                high_prices = df['high'].values.astype(np.float64)
                low_prices = df['low'].values.astype(np.float64)

                positions = main(ticker, close_prices, high_prices, low_prices, close_prices)
                if positions:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{now}] {ticker} positions:", positions)

            except Exception as e:
                print(f"Error with {coin['market']}: {e}")
                # 에러 발생 시 해당 코인은 건너뛰기
        time.sleep(60)  # 전체 배열 처리 후 10초 대기
