import yfinance as yf

class MarketData:
    def __init__(self):
        self.prices = None
    
    def fetch_data(self, tickers, start, end):
        try:
            self.prices = yf.download(tickers, start=start, end=end)["Adj Close"]
            return self.prices
        except Exception as e:
            return None
