"""
matching_engine.py

Core limit order book + matching engine for the simulated exchange.
Implements price-time priority matching (standard for real exchanges):
  - Orders are matched at the best available price
  - Among orders at the same price, earliest order fills first

This is the centerpiece of the project — it demonstrates understanding
of market microstructure, not just ML pattern matching.
"""

import heapq
import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


@dataclass
class Order:
    order_id: str
    user_id: str
    side: Side
    order_type: OrderType
    price: Optional[float]  # None for market orders
    quantity: float
    remaining: float = field(init=False)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        self.remaining = self.quantity


@dataclass
class Trade:
    trade_id: str
    buy_order_id: str
    sell_order_id: str
    price: float
    quantity: float
    timestamp: float = field(default_factory=time.time)
    buy_user_id: str = ""
    sell_user_id: str = ""


class OrderBook:
    """
    Maintains two priority heaps: bids (max-heap via negated price) and
    asks (min-heap). Ties broken by insertion order (time priority).
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._bid_heap: List = []  # (-price, seq, order)
        self._ask_heap: List = []  # (price, seq, order)
        self._seq_counter = itertools.count()
        self._trade_counter = itertools.count()
        self.trade_history: List[Trade] = []
        self.orders_by_id: dict = {}

    # ---------- public API ----------

    def submit_order(self, order: Order) -> List[Trade]:
        """Submit an order, attempt to match it, return list of resulting trades."""
        self.orders_by_id[order.order_id] = order
        trades = []

        if order.side == Side.BUY:
            trades = self._match_buy(order)
            if order.remaining > 0 and order.order_type == OrderType.LIMIT:
                heapq.heappush(
                    self._bid_heap,
                    (-order.price, next(self._seq_counter), order),
                )
        else:
            trades = self._match_sell(order)
            if order.remaining > 0 and order.order_type == OrderType.LIMIT:
                heapq.heappush(
                    self._ask_heap,
                    (order.price, next(self._seq_counter), order),
                )

        self.trade_history.extend(trades)
        return trades

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders_by_id.get(order_id)
        if order is None or order.remaining <= 0:
            return False
        order.remaining = 0  # lazy deletion; skipped during matching/depth calc
        return True

    def best_bid(self) -> Optional[float]:
        self._clean_heap(self._bid_heap)
        return -self._bid_heap[0][0] if self._bid_heap else None

    def best_ask(self) -> Optional[float]:
        self._clean_heap(self._ask_heap)
        return self._ask_heap[0][0] if self._ask_heap else None

    def depth(self, levels: int = 10) -> dict:
        """Return top N price levels on each side, aggregated quantity."""
        bids = self._aggregate(self._bid_heap, negate=True)[:levels]
        asks = self._aggregate(self._ask_heap, negate=False)[:levels]
        return {"bids": bids, "asks": asks}

    def last_price(self) -> Optional[float]:
        return self.trade_history[-1].price if self.trade_history else None

    # ---------- internal matching ----------

    def _match_buy(self, buy_order: Order) -> List[Trade]:
        trades = []
        while buy_order.remaining > 0:
            self._clean_heap(self._ask_heap)
            if not self._ask_heap:
                break
            best_ask_price, _, sell_order = self._ask_heap[0]
            if buy_order.order_type == OrderType.LIMIT and buy_order.price < best_ask_price:
                break  # no crossable price
            fill_qty = min(buy_order.remaining, sell_order.remaining)
            trade = Trade(
                trade_id=str(next(self._trade_counter)),
                buy_order_id=buy_order.order_id,
                sell_order_id=sell_order.order_id,
                price=best_ask_price,
                quantity=fill_qty,
                buy_user_id=buy_order.user_id,
                sell_user_id=sell_order.user_id,
            )
            trades.append(trade)
            buy_order.remaining -= fill_qty
            sell_order.remaining -= fill_qty
            if sell_order.remaining <= 0:
                heapq.heappop(self._ask_heap)
        return trades

    def _match_sell(self, sell_order: Order) -> List[Trade]:
        trades = []
        while sell_order.remaining > 0:
            self._clean_heap(self._bid_heap)
            if not self._bid_heap:
                break
            neg_price, _, buy_order = self._bid_heap[0]
            best_bid_price = -neg_price
            if sell_order.order_type == OrderType.LIMIT and sell_order.price > best_bid_price:
                break
            fill_qty = min(sell_order.remaining, buy_order.remaining)
            trade = Trade(
                trade_id=str(next(self._trade_counter)),
                buy_order_id=buy_order.order_id,
                sell_order_id=sell_order.order_id,
                price=best_bid_price,
                quantity=fill_qty,
                buy_user_id=buy_order.user_id,
                sell_user_id=sell_order.user_id,
            )
            trades.append(trade)
            sell_order.remaining -= fill_qty
            buy_order.remaining -= fill_qty
            if buy_order.remaining <= 0:
                heapq.heappop(self._bid_heap)
        return trades

    def _clean_heap(self, heap: List):
        """Pop off any orders at the top that were cancelled (remaining <= 0)."""
        while heap and heap[0][2].remaining <= 0:
            heapq.heappop(heap)

    def _aggregate(self, heap: List, negate: bool) -> List[dict]:
        levels = {}
        for entry in heap:
            price = -entry[0] if negate else entry[0]
            order = entry[2]
            if order.remaining <= 0:
                continue
            levels[price] = levels.get(price, 0) + order.remaining
        sorted_prices = sorted(levels.keys(), reverse=negate)
        return [{"price": p, "quantity": levels[p]} for p in sorted_prices]