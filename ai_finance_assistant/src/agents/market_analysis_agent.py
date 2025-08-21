import streamlit as st
from typing import Dict
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential
from ..workflow.state import AgentState
from ..rag.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class MarketAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=512)
        self.rag = RAGSystem([
            "fin_goal_planning_agent.json",
            "fin_market_analysis_agent.json",
            "fin_portfolio_analysis_agent.json",
            "fin_qna_agent_links.json",
            "fin_tax_education_agent.json",
        ])
        self.sp100_tickers = [
            'NVDA', 'MSFT', 'AAPL', 'GOOG', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'BRK.B',
            'WMT', 'JPM', 'ORCL', 'V', 'LLY', 'NFLX', 'MA', 'XOM', 'PLTR', 'COST',
            'JNJ', 'HD', 'PG', 'BAC', 'ABBV', 'CVX', 'KO', 'GE', 'TMUS', 'AMD',
            'CSCO', 'PM', 'WFC', 'UNH', 'MS', 'ABT', 'LIN', 'CRM', 'IBM', 'MCD',
            'GS', 'BX', 'AXP', 'RTX', 'DIS', 'T', 'PEP', 'MRK', 'INTU', 'CAT',
            'UBER', 'VZ', 'TMO', 'BLK', 'SCHW', 'GEV', 'ANET', 'NOW', 'BKNG', 'C',
            'BA', 'TXN', 'ISRG', 'SPGI', 'QCOM', 'AMGN', 'BSX', 'AMAT', 'GILD', 'ACN',
            'TJX', 'NEE', 'DHR', 'SYK', 'ADBE', 'MU', 'PGR', 'ETN', 'PFE', 'COF',
            'HON', 'LOW', 'DE', 'APH', 'LRCX', 'KKR', 'UNP', 'KLAC', 'ADP', 'CMCSA',
            'COP', 'MDT', 'PANW', 'SNPS', 'ADI', 'DASH', 'MO', 'NKE', 'WELL', 'CRWD'
        ]

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _fetch_stock_data(_self, ticker: str) -> Dict:
        try:
            data = yf.Ticker(ticker)
            info = data.info
            hist = data.history(period="1y")
            return {
                "ticker": ticker,
                "current_price": info.get("currentPrice", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "average_volume": info.get("averageVolume", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A") * 100 if info.get("dividendYield") else "N/A",
                "historical_data": hist
            }
        except Exception as e:
            return {"error": f"Failed to fetch data for {ticker}: {str(e)}"}

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def _generate_analysis(_self, ticker: str, data: Dict, query: str, context: str) -> str:
        prompt = f"""
        You are a market analyst. Provide a detailed analysis (2-3 paragraphs) of the stock, including:
        - Current performance based on price, market cap, P/E ratio, volume, and dividend yield
        - Historical price trends over the past year
        - Market trends and insights based on the provided context
        Data:
        - Ticker: {ticker}
        - Current Price: ${data.get('current_price', 'N/A')}
        - Market Cap: ${data.get('market_cap', 'N/A'):,.0f}
        - P/E Ratio: {data.get('pe_ratio', 'N/A')}
        - Average Volume: {data.get('average_volume', 'N/A'):,.0f}
        - Dividend Yield: {data.get('dividend_yield', 'N/A')}%
        - 52-Week High: ${data.get('fifty_two_week_high', 'N/A')}
        - 52-Week Low: ${data.get('fifty_two_week_low', 'N/A')}
        Query: {query}
        Context: {context}
        """
        try:
            analysis = _self.llm.invoke([
                SystemMessage(content="You are a knowledgeable market analyst providing detailed insights."),
                HumanMessage(content=prompt)
            ])
            return analysis.content
        except Exception as e:
            return f"Error generating analysis: {str(e)}"

    def execute(self, state: AgentState) -> AgentState:
        query = state["query"]
        ticker = None
        words = query.upper().split()
        #print(words)
        for word in words:
            if word in self.sp100_tickers:
                ticker = word
                break
        
        response = {}
        try:
            if not ticker:
                response = {"error": "Please include a valid S&P 100 ticker symbol in your query (e.g., AAPL, MSFT)."}
            else:
                # Fetch stock data with caching and retry
                response = self._fetch_stock_data(ticker)
                if "error" in response:
                    st.error(response["error"])
                else:
                    # Generate detailed analysis
                    context = "\n".join(self.rag.retrieve_context(query)) or "No additional context available."
                    response["analysis"] = self._generate_analysis(ticker, response, query, context)
        except Exception as e:
            st.error(f"Error processing market data: {str(e)}")
            response = {"error": f"Error processing market data: {str(e)}"}

        state["response"] = response
        return state
