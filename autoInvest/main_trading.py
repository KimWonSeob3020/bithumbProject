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

    prices = np.array(prices, dtype=np.float64)
    high = np.array(high, dtype=np.float64)
    low = np.array(low, dtype=np.float64)
    close = np.array(close, dtype=np.float64)

    signals = rsi_signals(prices, low=40, high=60)
    rsi = talib.RSI(prices, timeperiod=14)
    atr = talib.ATR(high, low, close, timeperiod=14)

    # tolerance를 40으로 확장
    support, resistance = find_support_resistance(prices, tolerance=40)

    # 다이버전스 window 파라미터 5로 축소
    bullish_diverge, bearish_diverge = rsi_divergence(prices, rsi, window=5)

    positions = []

    for i in range(len(signals)):
        signal = signals[i]

        # 신호만으로 매수/매도 실행, 가격 정보 포함 출력
        if signal == 1:
            exit_idx, exit_type = atr_stop_exit(prices, atr, i, rr_ratio=1.5)
            entry_price = prices[i]
            exit_price = prices[exit_idx] if exit_idx < len(prices) else None
            positions.append({'ticker': ticker, 'entry': i, 'exit': exit_idx, 'type': 'buy', 'exit_type': exit_type})
            print(f"Buy triggered at index {i} (price: {entry_price:.2f}), exit at {exit_idx} (price: {exit_price:.2f}) ({exit_type})")

        elif signal == -1:
            exit_idx, exit_type = atr_stop_exit(prices, atr, i, rr_ratio=1.5)
            entry_price = prices[i]
            exit_price = prices[exit_idx] if exit_idx < len(prices) else None
            positions.append({'ticker': ticker, 'entry': i, 'exit': exit_idx, 'type': 'sell', 'exit_type': exit_type})
            print(f"Sell triggered at index {i} (price: {entry_price:.2f}), exit at {exit_idx} (price: {exit_price:.2f}) ({exit_type})")

    save_positions(positions, 'positions.json')
    return positions


if __name__ == "__main__":
    bithumb_api = python_bithumb_api.Bithumb_api()
    select_coin_list = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-BCH', 'KRW-AVAX', 'KRW-DOGE', 'KRW-TRX', 'KRW-ADA', 'KRW-SOL']
    coin_list = bithumb_api.get_market_all()

    while True:
        for coin in select_coin_list:
            try:
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
                print(f"Error with {coin}: {e}")
            time.sleep(60)  # 전체 배열 처리 후 60초 대기
