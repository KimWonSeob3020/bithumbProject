#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bithumb v2.1.0 자동 RSI + 지지/저항 + 다이버전스 전략 봇 (DRY-RUN 기본)

📌 핵심 기능
- RSI(와일더) 기반 매수/매도 시그널
- 최근 스윙 기반 지지/저항 확인 및 캔들 돌파/이탈 체크
- RSI 다이버전스(강세/약세) 감지
- ATR 기반 손절/익절(R:R) 및 트레일링 스탑, 포지션 추적(JSON 저장)
- 빗썸 2.1.0 API로 실거래(시장가/지정가) 주문 (기본은 DRY_RUN=True)

⚠️ 중요
- 본 코드는 교육용 예시입니다. 실제 운용 전 모의 테스트/소액으로 충분히 점검하세요.
- 빗썸은 API로 OCO(동시 손절/익절) 주문을 제공하지 않으므로, 본 코드는 가격을 감시하여 시장가 청산(소프트 스탑)을 수행합니다.

필요 라이브러리
    pip install requests pandas numpy pyjwt python-dotenv

환경 변수(.env 또는 시스템 환경)
    BITHUMB_ACCESS_KEY=발급키
    BITHUMB_SECRET_KEY=발급시크릿

사용 예시
    python bithumb_rsi_sr_strategy.py

설정은 아래 CONFIG 섹션에서 변경하세요.
"""

from __future__ import annotations
import os
import time
import uuid
import hmac
import json
import math
import jwt  # pyjwt
import hashlib
import logging
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ==========================
# CONFIG (전략/거래 파라미터)
# ==========================
CONFIG = {
    "MARKET": "KRW-BTC",       # 예: KRW-BTC, KRW-ETH
    "CANDLE_UNIT": 5,           # 분봉 단위: 1/3/5/10/15/30/60/240
    "CANDLE_COUNT": 200,        # 요청 캔들 개수(최대 200)

    # RSI/ATR 파라미터
    "RSI_PERIOD": 14,           # RSI 기간
    "ATR_PERIOD": 14,           # ATR 기간

    # RSI 기준값
    "RSI_OVERSOLD": 30,
    "RSI_OVERBOUGHT": 70,
    "RSI_MIDLINE": 50,

    # 스윙 고저점(지지/저항) 탐색 폭 (왼쪽/오른쪽 캔들 수)
    "SWING_LEFT": 3,
    "SWING_RIGHT": 3,

    # 돌파/이탈 확증 여유 (레벨 대비 퍼센트)
    "BREAK_EPS": 0.001,  # 0.1%

    # 리스크 관리
    "DRY_RUN": True,         # True면 주문 전송 대신 로그만 출력
    "KRW_RISK_PER_TRADE": 0.05,  # 사용 가능 KRW의 5%만 매수 (시장가 매수는 금액 지정)
    "RR_TARGET": 2.0,       # 기본 R:R = 1:2
    "TRAIL_AFTER_R_MULT": 1.5,  # 1.5R 이익 도달 후 트레일링 스탑 활성화
    "MAX_OPEN_POSITIONS": 1,    # 동시 보유 포지션 수 제한

    # 로컬 파일 저장
    "STATE_FILE": "positions.json",

    # API 타임아웃/주기
    "POLL_SEC": 20,         # 루프 대기 초
    "TIMEZONE": "Asia/Seoul",
}

API_BASE = "https://api.bithumb.com"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rsi-sr-bot")

# ===============
# 데이터 클래스
# ===============
@dataclass
class Position:
    market: str
    side: str  # "long"만 사용 (현물 기준). 필요 시 "short"확장
    entry_price: float
    qty: float
    spent_krw: float
    stop_loss: float
    take_profit: float
    reason: str
    order_uuid: Optional[str]
    created_at: str
    trailing_active: bool = False

    def to_dict(self):
        return asdict(self)

# =============================
# 빗썸 API 클라이언트 (v2.1.0)
# =============================
class BithumbClient:
    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key
        self.session = requests.Session()

    # ---- 내부: 인증 토큰 생성 (파라미터 유무 따라 query_hash 포함) ----
    def _auth_headers(self, method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None) -> Dict[str, str]:
        # 빗썸 v2.1.0은 JWT(HS256) 인증을 사용. 파라미터가 있으면 query_hash(SHA512)를 페이로드에 포함.
        payload = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": round(time.time() * 1000),
        }
        # GET 쿼리 또는 POST 바디가 존재하면 query_hash 계산
        if params or body:
            if params:
                query = urlencode(params, doseq=True).encode()
            else:
                query = urlencode(body or {}, doseq=True).encode()
            h = hashlib.sha512()
            h.update(query)
            payload["query_hash"] = h.hexdigest()
            payload["query_hash_alg"] = "SHA512"
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    # ---- 공용 시세 ----
    def get_candles_minutes(self, market: str, unit: int, count: int = 200) -> List[dict]:
        url = f"{API_BASE}/v1/candles/minutes/{unit}"
        params = {"market": market, "count": min(200, int(count))}
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_ticker(self, market: str) -> dict:
        url = f"{API_BASE}/v1/ticker"
        params = {"market": market}
        r = self.session.get(url, params=params, timeout=5)
        r.raise_for_status()
        return r.json()

    # ---- 프라이빗 (서명 필요) ----
    def get_accounts(self) -> List[dict]:
        path = "/v1/accounts"
        headers = self._auth_headers("GET", path)
        r = self.session.get(f"{API_BASE}{path}", headers=headers, timeout=5)
        r.raise_for_status()
        return r.json()

    def get_orders_chance(self, market: str) -> dict:
        path = "/v1/orders/chance"
        params = {"market": market}
        headers = self._auth_headers("GET", path, params=params)
        r = self.session.get(f"{API_BASE}{path}", headers=headers, params=params, timeout=5)
        r.raise_for_status()
        return r.json()

    def place_order(self, market: str, side: str, ord_type: str, price: Optional[float] = None, volume: Optional[float] = None) -> dict:
        """
        side: "bid"(매수) | "ask"(매도)
        ord_type:
            - "limit"  : 지정가 (price, volume 필요)
            - "price"  : 시장가 매수 (price=KRW 총액 필요, volume=None)
            - "market" : 시장가 매도 (volume 필요, price=None)
        참고: 빗썸 문서 정의에 맞춤.
        """
        path = "/v1/orders"
        body: Dict[str, str] = {"market": market, "side": side, "ord_type": ord_type}
        if price is not None:
            body["price"] = str(price)
        if volume is not None:
            body["volume"] = str(volume)
        headers = self._auth_headers("POST", path, body=body)
        r = self.session.post(f"{API_BASE}{path}", headers=headers, json=body, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_order(self, uuid_str: str) -> dict:
        path = "/v1/order"
        params = {"uuid": uuid_str}
        headers = self._auth_headers("GET", path, params=params)
        r = self.session.get(f"{API_BASE}{path}", headers=headers, params=params, timeout=5)
        r.raise_for_status()
        return r.json()

    def list_orders(self, market: str, state: str = "wait", limit: int = 100) -> List[dict]:
        path = "/v1/orders"
        params = {"market": market, "state": state, "limit": limit}
        headers = self._auth_headers("GET", path, params=params)
        r = self.session.get(f"{API_BASE}{path}", headers=headers, params=params, timeout=5)
        r.raise_for_status()
        return r.json()

    def cancel_order(self, uuid_str: str) -> dict:
        path = "/v1/order"
        params = {"uuid": uuid_str}
        headers = self._auth_headers("DELETE", path, params=params)
        r = self.session.delete(f"{API_BASE}{path}", headers=headers, params=params, timeout=5)
        r.raise_for_status()
        return r.json()

# ===================
# 유틸 / 보조 함수
# ===================

def load_env_keys() -> Tuple[str, str]:
    load_dotenv()
    access = os.getenv("BITHUMB_ACCESS_KEY", "")
    secret = os.getenv("BITHUMB_SECRET_KEY", "")
    if not access or not secret:
        logger.warning("API 키가 설정되지 않았습니다. .env 또는 환경변수를 확인하세요.")
    return access, secret


def round_down_to_unit(value: float, unit: float) -> float:
    if unit <= 0:
        return value
    return math.floor(value / unit) * unit


def to_dataframe(candles: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    # 컬럼 표준화
    df = df.rename(
        columns={
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
            "candle_date_time_kst": "time_kst",
        }
    )
    # 시간 오름차순 정렬
    df["time_kst"] = pd.to_datetime(df["time_kst"])  # KST 문자열 -> datetime
    df = df.sort_values("time_kst").reset_index(drop=True)
    return df[["time_kst", "open", "high", "low", "close", "volume"]]


# ============
# 지표 계산기
# ============

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (EMA 기반)"""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1/period, adjust=False).mean()


# =========================
# 스윙 고저/지지저항 & 돌파
# =========================

def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    프랙탈 방식 스윙 고점/저점 마킹
    - 스윙고점: 해당 캔들의 high가 좌우 N개보다 모두 높음
    - 스윙저점: 해당 캔들의 low가 좌우 N개보다 모두 낮음
    반환: (swing_high: bool Series, swing_low: bool Series)
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        if highs[i] == max(highs[i - left: i + right + 1]):
            swing_high[i] = True
        if lows[i] == min(lows[i - left: i + right + 1]):
            swing_low[i] = True
    return pd.Series(swing_high, index=df.index), pd.Series(swing_low, index=df.index)


def nearest_sr_levels(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series, ref_price: float) -> Tuple[Optional[float], Optional[float]]:
    """최근 스윙 기준, ref_price 주변의 가장 가까운 저항/지지 레벨을 찾음"""
    highs = df.loc[swing_high, "high"].values[::-1]  # 최근순
    lows = df.loc[swing_low, "low"].values[::-1]
    res = next((h for h in highs if h >= ref_price), None)
    sup = next((l for l in lows if l <= ref_price), None)
    return res, sup


def is_breakout(close: float, level: float, eps: float = 0.001) -> bool:
    return level is not None and close > level * (1 + eps)


def is_breakdown(close: float, level: float, eps: float = 0.001) -> bool:
    return level is not None and close < level * (1 - eps)


# =========================
# RSI 다이버전스 감지
# =========================

def last_two_swings(values: pd.Series, flags: pd.Series, kind: str) -> Optional[Tuple[Tuple[int, float], Tuple[int, float]]]:
    """
    kind: "high" 또는 "low"
    flags(True)인 최근 두 지점의 (index, value)를 반환
    """
    idxs = list(values[flags].index)
    if len(idxs) < 2:
        return None
    i2, i1 = idxs[-2], idxs[-1]  # 과거, 최근
    v2 = float(values.loc[i2])
    v1 = float(values.loc[i1])
    return (i2, v2), (i1, v1)


def detect_divergence(df: pd.DataFrame, rsi_series: pd.Series, swing_high: pd.Series, swing_low: pd.Series) -> Tuple[bool, bool, str]:
    """강세/약세 다이버전스 판별 및 설명 리턴 (bullish, bearish, reason)"""
    reason = []
    bullish = False
    bearish = False

    # 강세: 가격 LL, RSI HL (저점 기준)
    lows_pair = last_two_swings(df["low"], swing_low, kind="low")
    rsi_lows_pair = last_two_swings(rsi_series, swing_low, kind="low")
    if lows_pair and rsi_lows_pair:
        (_, p2), (_, p1) = lows_pair
        (_, r2), (_, r1) = rsi_lows_pair
        if p1 < p2 and r1 > r2:
            bullish = True
            reason.append(f"Bullish div: price LL({p2:.0f}->{p1:.0f}), RSI HL({r2:.1f}->{r1:.1f}))")

    # 약세: 가격 HH, RSI LH (고점 기준)
    highs_pair = last_two_swings(df["high"], swing_high, kind="high")
    rsi_highs_pair = last_two_swings(rsi_series, swing_high, kind="high")
    if highs_pair and rsi_highs_pair:
        (_, p2), (_, p1) = highs_pair
        (_, r2), (_, r1) = rsi_highs_pair
        if p1 > p2 and r1 < r2:
            bearish = True
            reason.append(f"Bearish div: price HH({p2:.0f}->{p1:.0f}), RSI LH({r2:.1f}->{r1:.1f}))")

    return bullish, bearish, "; ".join(reason)


# ========================
# 시그널 생성 로직 (현물)
# ========================

def generate_signal(df: pd.DataFrame, cfg: dict) -> Tuple[str, str, Dict[str, float]]:
    """
    반환: (signal, reason, context)
        - signal: "LONG", "EXIT", "HOLD"
        - reason: 사람이 읽을 수 있는 문자열
        - context: 보조 정보(dict)
    """
    rsi_series = rsi(df["close"], cfg["RSI_PERIOD"]).fillna(50)
    atr_series = atr(df, cfg["ATR_PERIOD"]).fillna(method="bfill").fillna(0)

    swing_high, swing_low = find_swings(df, cfg["SWING_LEFT"], cfg["SWING_RIGHT"])

    close = float(df["close"].iloc[-1])
    open_ = float(df["open"].iloc[-1])
    rsi_now = float(rsi_series.iloc[-1])
    atr_now = float(atr_series.iloc[-1])

    res, sup = nearest_sr_levels(df, swing_high, swing_low, close)
    bullish_div, bearish_div, div_reason = detect_divergence(df, rsi_series, swing_high, swing_low)

    reason = []
    ctx = {"close": close, "rsi": rsi_now, "atr": atr_now, "res": res or np.nan, "sup": sup or np.nan}

    # --- 진입 조건 (LONG) ---
    long_ok = False

    # (1) RSI가 상향(50 상회) + 최근 저항 돌파 종가 확정
    if rsi_now > cfg["RSI_MIDLINE"] and res is not None and is_breakout(close, res, cfg["BREAK_EPS"]):
        long_ok = True
        reason.append(f"RSI>{cfg['RSI_MIDLINE']} & Resistance({res:.0f}) breakout by close {close:.0f}")

    # (2) 혹은 강세 다이버전스 발생 & 강한 양봉(종가>시가)으로 전환
    if bullish_div and close > open_:
        long_ok = True
        reason.append("Bullish divergence + bullish candle")

    # (3) 또한, 지지 확인: 종가가 최근 지지(sup) 위에 위치
    if sup is not None and close > sup * (1 + cfg["BREAK_EPS"]):
        reason.append(f"Support held above {sup:.0f}")
    else:
        # 지지가 불확실하면 진입 강도 낮춤
        pass

    # --- 청산 조건 (EXIT) ---
    exit_ok = False
    # (a) RSI가 과매수에서 하향 이탈
    if rsi_now < cfg["RSI_OVERBOUGHT"] and rsi_now < cfg["RSI_MIDLINE"] and bearish_div:
        exit_ok = True
        reason.append("Bearish divergence forming with RSI loss of momentum")

    # (b) 지지 이탈
    if sup is not None and is_breakdown(close, sup, cfg["BREAK_EPS"]):
        exit_ok = True
        reason.append(f"Breakdown below support {sup:.0f}")

    # (c) 약세 장대음봉
    if close < open_ and (open_ - close) > 0.8 * atr_now:
        exit_ok = True
        reason.append("Large bearish candle vs ATR")

    # 최종 판정: 롱 진입이 더 강하면 LONG, 그렇지 않고 청산 신호면 EXIT
    if long_ok and not exit_ok:
        return "LONG", "; ".join(reason + ([div_reason] if div_reason else [])), ctx
    elif exit_ok and not long_ok:
        return "EXIT", "; ".join(reason + ([div_reason] if div_reason else [])), ctx
    else:
        return "HOLD", "; ".join(reason + ([div_reason] if div_reason else [])), ctx


# =============================
# 포지션 저장/로딩 (로컬 JSON)
# =============================
class PositionStore:
    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, Position] = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._data = {k: Position(**v) for k, v in raw.items()}
        else:
            self._data = {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self._data.items()}, f, ensure_ascii=False, indent=2)

    def get(self, market: str) -> Optional[Position]:
        return self._data.get(market)

    def set(self, pos: Position):
        self._data[pos.market] = pos
        self.save()

    def delete(self, market: str):
        if market in self._data:
            del self._data[market]
            self.save()


# ======================
# 주문/체결 보조 로직
# ======================

def calc_risk_kwargs(client: BithumbClient, cfg: dict) -> Tuple[float, float, float]:
    """주문 가능 정보와 KRW 잔고를 조회하여 시장가 매수 금액 제한/반올림 적용"""
    chance = client.get_orders_chance(cfg["MARKET"])  # 수수료, min_total, price_unit 등
    # 잔고 조회
    accts = client.get_accounts()
    krw_bal = 0.0
    for a in accts:
        if a.get("currency") == "KRW":
            krw_bal = float(a.get("balance", 0.0))
            break

    min_total = float(chance["market"]["bid"]["min_total"])  # 최소 주문 금액
    price_unit = float(chance["market"]["bid"]["price_unit"])  # 금액 단위

    max_spend = krw_bal * cfg["KRW_RISK_PER_TRADE"]
    spend = max(min_total, max_spend)
    spend = round_down_to_unit(spend, price_unit)

    return spend, min_total, price_unit


def place_market_buy(client: BithumbClient, cfg: dict, spend_krw: float) -> dict:
    """빗썸 규격에 맞춘 시장가 매수: ord_type="price", price=KRW 총액"""
    if cfg["DRY_RUN"]:
        logger.info(f"[DRY_RUN] 시장가 매수 금액 KRW {spend_krw:.0f}")
        return {"uuid": None, "dry_run": True}
    return client.place_order(cfg["MARKET"], side="bid", ord_type="price", price=spend_krw)


def place_market_sell(client: BithumbClient, cfg: dict, volume: float) -> dict:
    """빗썸 규격에 맞춘 시장가 매도: ord_type="market", volume=수량"""
    if cfg["DRY_RUN"]:
        logger.info(f"[DRY_RUN] 시장가 매도 수량 {volume}")
        return {"uuid": None, "dry_run": True}
    return client.place_order(cfg["MARKET"], side="ask", ord_type="market", volume=volume)


# ============================
# 손절/익절/트레일 관리 로직
# ============================

def build_sl_tp(entry: float, df: pd.DataFrame, cfg: dict, near_support: Optional[float], near_resistance: Optional[float]) -> Tuple[float, float]:
    """ATR와 S/R을 함께 고려하여 SL/TP 산출"""
    a = float(atr(df, cfg["ATR_PERIOD"]).iloc[-1])
    # SL: 지지 아래 또는 ATR*1.5 아래 중 더 낮은 값
    sl1 = entry - 1.5 * a
    sl2 = near_support * (1 - cfg["BREAK_EPS"]) if near_support else entry - 1.0 * a
    sl = min(sl1, sl2)

    # TP: R:R 기준 또는 근처 저항
    tp_rr = entry + cfg["RR_TARGET"] * (entry - sl)
    tp_res = near_resistance * (1 - cfg["BREAK_EPS"]) if near_resistance else tp_rr
    # 더 보수적으로 가까운 값을 채택
    tp = min(tp_rr, tp_res) if near_resistance else tp_rr
    return sl, tp


def maybe_update_trailing(pos: Position, last_price: float, cfg: dict):
    """이익이 일정 배수(R)에 도달하면 SL을 끌어올리는 간단한 트레일"""
    r = pos.entry_price - pos.stop_loss
    if r <= 0:
        return
    profit = last_price - pos.entry_price
    if profit >= cfg["TRAIL_AFTER_R_MULT"] * r:
        # 최근 가격 기준 1*ATR 뒤로 끌어올리는 보수적 방식
        pos.trailing_active = True
        # 절대 이동(예시): 익절 절반 수준까지 SL 상향
        new_sl = pos.entry_price  # BE로 이동
        if new_sl > pos.stop_loss:
            logger.info(f"트레일링 발동: SL {pos.stop_loss:.0f} -> {new_sl:.0f}")
            pos.stop_loss = new_sl


# ================================
# 메인 루프: 데이터 수집/판단/주문
# ================================

def main():
    access, secret = load_env_keys()
    client = BithumbClient(access, secret)
    store = PositionStore(CONFIG["STATE_FILE"])

    market = CONFIG["MARKET"]
    unit = CONFIG["CANDLE_UNIT"]

    while True:
        try:
            candles = client.get_candles_minutes(market, unit, CONFIG["CANDLE_COUNT"])
            df = to_dataframe(candles)
            sig, why, ctx = generate_signal(df, CONFIG)
            close = ctx["close"]

            pos = store.get(market)

            if pos is None:
                # 포지션 없음 -> 진입만 고려
                if sig == "LONG":
                    spend, min_total, unit_krw = calc_risk_kwargs(client, CONFIG)
                    # 손절/익절 산출
                    swing_high, swing_low = find_swings(df, CONFIG["SWING_LEFT"], CONFIG["SWING_RIGHT"])
                    res, sup = nearest_sr_levels(df, swing_high, swing_low, close)
                    sl, tp = build_sl_tp(close, df, CONFIG, sup, res)

                    logger.info(f"LONG 신호: {why}")
                    logger.info(f"진입가~{close:.0f}, SL~{sl:.0f}, TP~{tp:.0f}, 매수금액 KRW {spend:.0f}")

                    # 주문 전송 (시장가 매수는 KRW 총액을 price에)
                    order_resp = place_market_buy(client, CONFIG, spend)
                    order_uuid = order_resp.get("uuid") if order_resp else None

                    # 체결 수량 추정: 시장가 매수 즉시 체결 가정 -> 수량=KRW/가격
                    qty_est = spend / close if close > 0 else 0

                    new_pos = Position(
                        market=market,
                        side="long",
                        entry_price=close,
                        qty=qty_est,
                        spent_krw=spend,
                        stop_loss=sl,
                        take_profit=tp,
                        reason=why,
                        order_uuid=order_uuid,
                        created_at=dt.datetime.now().isoformat(),
                    )
                    store.set(new_pos)
                else:
                    logger.info(f"HOLD: {why} | close={close:.0f} RSI={ctx['rsi']:.1f}")

            else:
                # 포지션 보유 중 -> 손절/익절/트레일 체크
                maybe_update_trailing(pos, close, CONFIG)
                if close <= pos.stop_loss:
                    logger.info(f"손절 조건 충족: price {close:.0f} <= SL {pos.stop_loss:.0f}")
                    sell_resp = place_market_sell(client, CONFIG, pos.qty)
                    logger.info(f"청산(손절) 주문 응답: {sell_resp}")
                    store.delete(market)
                elif close >= pos.take_profit:
                    logger.info(f"익절 조건 충족: price {close:.0f} >= TP {pos.take_profit:.0f}")
                    sell_resp = place_market_sell(client, CONFIG, pos.qty)
                    logger.info(f"청산(익절) 주문 응답: {sell_resp}")
                    store.delete(market)
                elif sig == "EXIT":
                    logger.info(f"전략 청산 신호: {why}")
                    sell_resp = place_market_sell(client, CONFIG, pos.qty)
                    logger.info(f"청산(전략) 주문 응답: {sell_resp}")
                    store.delete(market)
                else:
                    # 유지
                    store.set(pos)  # SL 조정 반영
                    logger.info(f"포지션 유지: entry={pos.entry_price:.0f}, SL={pos.stop_loss:.0f}, TP={pos.take_profit:.0f} | close={close:.0f}")

        except requests.HTTPError as e:
            logger.error(f"HTTP 오류: {e} | 응답: {e.response.text if hasattr(e, 'response') and e.response is not None else ''}")
        except Exception as e:
            logger.exception(f"예상치 못한 오류: {e}")

        time.sleep(CONFIG["POLL_SEC"])  # 다음 폴링까지 대기


if __name__ == "__main__":
    main()