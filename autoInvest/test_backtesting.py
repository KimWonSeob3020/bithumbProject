import pandas as pd
from main_trading import main

def backtest():
    # CSV 데이터 로드
    data = pd.read_csv('upbit_1week_5min.csv')

    # 필요한 컬럼 리스트 변환
    prices = data['close'].tolist()
    high = data['high'].tolist()
    low = data['low'].tolist()
    close = data['close'].tolist()

    # 자동매매 메인 함수 실행
    positions = main('KRW-BTC', prices, high, low, close)
    #print(f"prices: {prices}, high: {high}, low: {low}, close: {close}")

    # 포지션 결과 출력
    for pos in positions:
        print(f"Entry: {pos['entry']}, Exit: {pos['exit']}, Type: {pos['type']}, Exit Type: {pos['exit_type']}")

if __name__ == '__main__':
    backtest()
