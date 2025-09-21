import os
from dotenv import load_dotenv
import python_bithumb


class Bithumb_api:

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("BITHUMB_API_KEY")
        secret_key = os.getenv("BITHUMB_SECRET_KEY")
        self.private_bithumb = python_bithumb.Bithumb(api_key, secret_key)
        self.public_bithumb = python_bithumb.public_api;

    # 구매 : order_info = Bithumb_api.buy_market_order("KRW-BTC", 10000)
    # {'uuid': 'C0101000002475487804', 'side': 'bid', 'ord_type': 'price', 'price': '10000', 'state': 'wait', 'market': 'KRW-BTC', 'created_at': '2025-09-18T14:43:21+09:00', 'reserved_fee': '25', 'remaining_fee': '25', 'paid_fee': '0', 'locked': '10026', 'executed_volume': '0', 'trades_count': 0}
    def buy_market_order(self, ticker: str, krw_amount: float):
        order_info = self.private_bithumb.buy_market_order("KRW-BTC", 10000)
        return order_info

    # 잔액조회 : balance = bithumb.get_balance("BTC")
    # 6.146e-05
    def get_balance(self, currency: str) -> float:
        balance = self.private_bithumb.get_balance(currency)
        return balance

    # 판매 : order_info = bithumb.sell_market_order("KRW-BTC", 6.146e-05)
    # {'uuid': 'C0101000002475488763', 'side': 'ask', 'ord_type': 'market', 'state': 'wait', 'market': 'KRW-BTC', 'created_at': '2025-09-18T14:44:51+09:00', 'volume': '0.00006146', 'remaining_volume': '0.00006146', 'reserved_fee': '0', 'remaining_fee': '0', 'paid_fee': '0', 'locked': '0.00006146', 'executed_volume': '0', 'trades_count': 0}
    def sell_market_order(self, ticker: str, volume: float):
        order_info = self.private_bithumb.sell_market_order(ticker, volume)
        return order_info

    def get_ohlcv(self, ticker: str, interval: str = "day", count: int = 200, period: float = 0.1, to: str = None):
        ohlcv_info = self.public_bithumb.get_ohlcv(ticker, interval, count, period, to)
        return ohlcv_info

    def get_market_all(self):
        market_all_list = self.public_bithumb.get_market_all();
        return market_all_list