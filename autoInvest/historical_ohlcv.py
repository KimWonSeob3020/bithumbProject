import requests
import pandas as pd
import time


def fetch_minute_candles(ticker='KRW-BTC', unit=5, count=200, to=None):
    """
    업비트 5분봉 캔들 데이터 조회
    :param ticker: 마켓 코드 (ex. KRW-BTC)
    :param unit: 분 단위 (1,3,5,10,15,30,60,240)
    :param count: 한번 조회 최대 개수 (최대 200)
    :param to: 종료 시점 (ISO8601 문자열 or None)
    :return: JSON 리스트
    """
    url = f'https://api.upbit.com/v1/candles/minutes/{unit}'
    params = {'market': ticker, 'count': count}
    if to is not None:
        params['to'] = to
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def fetch_last_week_5m_candles(ticker='KRW-BTC'):
    all_candles = []
    count_per_call = 200
    to = None

    # 7일 * 24시간 * 60분 / 5분 = 2016개 필요
    total_required = 20000

    while len(all_candles) < total_required:
        count = min(count_per_call, total_required - len(all_candles))
        candles = fetch_minute_candles(ticker=ticker, unit=5, count=count, to=to)
        if not candles:
            break
        all_candles.extend(candles)
        # 가장 오래된 캔들의 'candle_date_time_utc' 을 to로 지정
        to = candles[-1]['candle_date_time_utc']

        # API 쿨다운 제어
        time.sleep(0.3)

    # DataFrame으로 정리, 열 이름 조정
    df = pd.DataFrame(all_candles)
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'])
    df.sort_values(by='candle_date_time_kst', inplace=True)
    df = df[
        ['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    return df.reset_index(drop=True)


def save_ohlcv_to_csv(filename='upbit_1week_5min.csv'):
    df = fetch_last_week_5m_candles()
    df.to_csv(filename, index=False)
    print(f"{filename} 생성 완료")


if __name__ == "__main__":
    save_ohlcv_to_csv()
