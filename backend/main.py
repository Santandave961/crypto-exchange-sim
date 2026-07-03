"""
main.py

FastAPI application exposing the simulated exchange:
  - REST endpoints to place/cancel orders, view order book depth, trade history
  - WebSocket endpoint for live order book + trade updates
  - Wallet simulation (mock balances only — no real money/crypto moves)
  - Fraud alerts surfaced alongside trading activity

Run with:
    uvicorn main:app --reload --port 8000
"""

import uuid
from collections import defaultdict
from typing import Dict, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from fraud_detection import FraudDetector
from matching_engine import Order, OrderBook, OrderType, Side
from price_feed import get_live_prices

app = FastAPI(title="Simulated Crypto Exchange API")

SYMBOLS = ["BTC", "ETH", "USDT"]
order_books: Dict[str, OrderBook] = {sym: OrderBook(sym) for sym in SYMBOLS}
fraud_detector = FraudDetector()

# mock wallets: user_id -> {symbol: balance}
STARTING_BALANCE = {"USD": 100_000.0, "BTC": 5.0, "ETH": 50.0, "USDT": 100_000.0}
wallets: Dict[str, Dict[str, float]] = defaultdict(lambda: dict(STARTING_BALANCE))

connected_clients: List[WebSocket] = []


# ---------- request/response models ----------

class OrderRequest(BaseModel):
    user_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: float | None = None  # required for limit orders


class CancelRequest(BaseModel):
    symbol: str
    order_id: str
    user_id: str


# ---------- REST endpoints ----------

@app.get("/")
def root():
    return {"status": "ok", "symbols": SYMBOLS}


@app.get("/prices")
def prices():
    """Live reference prices pulled from CoinGecko."""
    return get_live_prices(SYMBOLS)


@app.get("/orderbook/{symbol}")
def orderbook(symbol: str, levels: int = 10):
    book = _get_book(symbol)
    return {
        "symbol": symbol.upper(),
        "depth": book.depth(levels),
        "last_price": book.last_price(),
    }


@app.get("/wallet/{user_id}")
def wallet(user_id: str):
    return wallets[user_id]


@app.post("/order")
async def place_order(req: OrderRequest):
    book = _get_book(req.symbol)

    if req.order_type == OrderType.LIMIT and req.price is None:
        raise HTTPException(400, "Limit orders require a price")

    order = Order(
        order_id=str(uuid.uuid4()),
        user_id=req.user_id,
        side=req.side,
        order_type=req.order_type,
        price=req.price,
        quantity=req.quantity,
    )

    fraud_detector.record_order(req.user_id, order.order_id, req.quantity, req.price or 0)

    trades = book.submit_order(order)

    for trade in trades:
        fraud_detector.record_trade(trade.buy_user_id, trade.sell_user_id, order.order_id)
        _settle_trade(req.symbol, trade)

    await _broadcast_update(req.symbol)

    return {
        "order_id": order.order_id,
        "status": "filled" if order.remaining == 0 else "partially_filled" if trades else "open",
        "remaining": order.remaining,
        "trades": [t.__dict__ for t in trades],
        "fraud_alerts": [a.__dict__ for a in fraud_detector.recent_alerts(3)],
    }


@app.post("/cancel")
async def cancel_order(req: CancelRequest):
    book = _get_book(req.symbol)
    success = book.cancel_order(req.order_id)
    if success:
        fraud_detector.record_cancel(req.user_id, req.order_id)
    return {"cancelled": success}


@app.get("/trades/{symbol}")
def trade_history(symbol: str, limit: int = 50):
    book = _get_book(symbol)
    return [t.__dict__ for t in book.trade_history[-limit:][::-1]]


@app.get("/fraud-alerts")
def fraud_alerts(limit: int = 20):
    return [a.__dict__ for a in fraud_detector.recent_alerts(limit)]


# ---------- WebSocket ----------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive; client doesn't need to send anything meaningful
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


async def _broadcast_update(symbol: str):
    book = _get_book(symbol)
    payload = {
        "type": "orderbook_update",
        "symbol": symbol.upper(),
        "depth": book.depth(10),
        "last_price": book.last_price(),
    }
    dead = []
    for client in connected_clients:
        try:
            await client.send_json(payload)
        except Exception:
            dead.append(client)
    for d in dead:
        connected_clients.remove(d)


# ---------- helpers ----------

def _get_book(symbol: str) -> OrderBook:
    sym = symbol.upper()
    if sym not in order_books:
        raise HTTPException(404, f"Unknown symbol: {symbol}")
    return order_books[sym]


def _settle_trade(symbol: str, trade):
    """Move mock balances between buyer/seller wallets. No real funds involved."""
    sym = symbol.upper()
    cost = trade.price * trade.quantity

    buyer = wallets[trade.buy_user_id]
    seller = wallets[trade.sell_user_id]

    buyer["USD"] = buyer.get("USD", 0) - cost
    buyer[sym] = buyer.get(sym, 0) + trade.quantity

    seller["USD"] = seller.get("USD", 0) + cost
    seller[sym] = seller.get(sym, 0) - trade.quantity