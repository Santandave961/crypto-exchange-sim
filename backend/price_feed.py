"""
price_feed.py

Pulls live spot prices from the CoinGecko public API so the simulated
order book seeds/resets around real market prices instead of fake data.
No API key required (public endpoint, rate-limited).
"""

import requests

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

SYMBOL_TO_ID = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "USDT": "tether",
}


def get_live_price(symbol: str, vs_currency: str = "usd") -> float:
    """Fetch the current live price for a symbol (e.g. 'BTC')."""
    coin_id = SYMBOL_TO_ID.get(symbol.upper())
    if coin_id is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    params = {"ids": coin_id, "vs_currencies": vs_currency}
    response = requests.get(COINGECKO_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data[coin_id][vs_currency]


def get_live_prices(symbols: list, vs_currency: str = "usd") -> dict:
    """Fetch multiple symbols in a single call (preferred to avoid rate limits)."""
    ids = [SYMBOL_TO_ID[s.upper()] for s in symbols if s.upper() in SYMBOL_TO_ID]
    params = {"ids": ",".join(ids), "vs_currencies": vs_currency}
    response = requests.get(COINGECKO_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    result = {}
    for symbol in symbols:
        coin_id = SYMBOL_TO_ID.get(symbol.upper())
        if coin_id and coin_id in data:
            result[symbol.upper()] = data[coin_id][vs_currency]
    return result


if __name__ == "__main__":
    prices = get_live_prices(["BTC", "ETH", "USDT"])
    for sym, price in prices.items():
        print(f"{sym}: ${price:,.2f}")