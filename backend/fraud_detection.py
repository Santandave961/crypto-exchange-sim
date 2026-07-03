"""
fraud_detection.py

Anomaly detection layer for the exchange, reusing the Isolation Forest
approach from AMLGuard AI. Flags suspicious trading patterns:

  - Wash trading: same user (or paired accounts) buying and selling
    the same asset repeatedly to fake volume
  - Rapid-fire order cancellation: placing and cancelling orders in
    quick succession (spoofing / quote stuffing pattern)
  - Abnormal order size relative to a user's own trading history
  - Abnormal order frequency in a short time window

Each flagged event includes a human-readable reason code, mirroring
the "explainability" pitch used for AMLGuard AI / SHAP-based tooling.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class FraudAlert:
    user_id: str
    order_id: str
    reason: str
    severity: str  # "low" | "medium" | "high"
    timestamp: float


class FraudDetector:
    def __init__(self, window_seconds: int = 60, cancel_threshold: int = 5,
                 wash_trade_threshold: int = 3):
        self.window_seconds = window_seconds
        self.cancel_threshold = cancel_threshold
        self.wash_trade_threshold = wash_trade_threshold

        # rolling per-user activity logs
        self.order_events: Dict[str, Deque[float]] = defaultdict(deque)
        self.cancel_events: Dict[str, Deque[float]] = defaultdict(deque)
        self.trade_pairs: Dict[str, Deque[tuple]] = defaultdict(deque)  # user -> (counterparty, ts)
        self.order_sizes: Dict[str, List[float]] = defaultdict(list)

        self.alerts: List[FraudAlert] = []

        # Isolation Forest trained lazily once enough order-size history exists
        self._iforest_models: Dict[str, IsolationForest] = {}

    # ---------- event ingestion ----------

    def record_order(self, user_id: str, order_id: str, quantity: float, price: float):
        now = time.time()
        self._trim(self.order_events[user_id], now)
        self.order_events[user_id].append(now)
        self.order_sizes[user_id].append(quantity)

        # frequency check
        if len(self.order_events[user_id]) > 20:
            self._raise(user_id, order_id, "High order frequency in short window", "medium")

        # size anomaly check (needs history)
        if len(self.order_sizes[user_id]) >= 10:
            if self._is_size_anomalous(user_id, quantity):
                self._raise(user_id, order_id, "Order size is a statistical outlier vs user history", "medium")

    def record_cancel(self, user_id: str, order_id: str):
        now = time.time()
        self._trim(self.cancel_events[user_id], now)
        self.cancel_events[user_id].append(now)
        if len(self.cancel_events[user_id]) >= self.cancel_threshold:
            self._raise(user_id, order_id, "Rapid order cancellation pattern (possible spoofing)", "high")

    def record_trade(self, buy_user_id: str, sell_user_id: str, order_id: str):
        now = time.time()
        self._trim_pairs(self.trade_pairs[buy_user_id], now)
        self.trade_pairs[buy_user_id].append((sell_user_id, now))
        recent_counterparties = [c for c, _ in self.trade_pairs[buy_user_id]]
        if recent_counterparties.count(sell_user_id) >= self.wash_trade_threshold:
            self._raise(buy_user_id, order_id,
                        f"Repeated trading with same counterparty ({sell_user_id}) — possible wash trading",
                        "high")

    # ---------- internals ----------

    def _is_size_anomalous(self, user_id: str, quantity: float) -> bool:
        history = np.array(self.order_sizes[user_id][-50:]).reshape(-1, 1)
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        model.fit(history)
        pred = model.predict([[quantity]])
        return pred[0] == -1

    def _trim(self, dq: Deque[float], now: float):
        while dq and now - dq[0] > self.window_seconds:
            dq.popleft()

    def _trim_pairs(self, dq: Deque[tuple], now: float):
        while dq and now - dq[0][1] > self.window_seconds:
            dq.popleft()

    def _raise(self, user_id: str, order_id: str, reason: str, severity: str):
        alert = FraudAlert(user_id=user_id, order_id=order_id, reason=reason,
                            severity=severity, timestamp=time.time())
        self.alerts.append(alert)

    def recent_alerts(self, limit: int = 20) -> List[FraudAlert]:
        return self.alerts[-limit:][::-1]