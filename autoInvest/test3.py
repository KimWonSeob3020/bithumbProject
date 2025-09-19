import os
from dotenv import load_dotenv
import python_bithumb
import python_bithumb_api


bithumb_api = python_bithumb_api.Bithumb_api()

#구매 시작
# KRW-BTC를 10,000원어치 시장가 매수
#order_info = bithumb_api.buy_market_order("KRW-BTC", 10000)
#print(order_info)
#구매 끝


#시장가 전량 매도

# 보유한 BTC 수량 조회
#balance = bithumb_api.get_balance("BTC")
#print(balance)

# 보유한 BTC 전량 시장가 매도
#order_info = bithumb_api.sell_market_order("KRW-BTC", balance)
#print(order_info)
#시장가 전량 매도

order_info = bithumb_api.get_ohlcv("KRW-BTC", "minute5");
print(order_info.head())