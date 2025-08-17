import pandas as pd

class Portfolio:
    def __init__(self, file):
        self.data = self._load_portfolio(file)
    
    def _load_portfolio(self, file):
        try:
            df = pd.read_csv(file)
            required_columns = ["Ticker", "Quantity"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError("CSV must contain 'Ticker' and 'Quantity' columns")
            df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
            if df["Quantity"].isnull().any():
                raise ValueError("Invalid quantities in CSV")
            return df
        except Exception as e:
            return None
    
    def get_tickers(self):
        return self.data["Ticker"].tolist() if self.data is not None else []

    def get_composition(self, prices):
        if self.data is None or prices is None:
            return None
        latest_prices = prices.iloc[-1]
        composition = pd.DataFrame({
            "Ticker": self.data["Ticker"],
            "Quantity": self.data["Quantity"],
            "Latest Price": [latest_prices[ticker] for ticker in self.data["Ticker"]],
            "Value": [latest_prices[ticker] * qty for ticker, qty in zip(self.data["Ticker"], self.data["Quantity"])],
        })
        composition["Weight (%)"] = composition["Value"] / composition["Value"].sum() * 100
        return composition
