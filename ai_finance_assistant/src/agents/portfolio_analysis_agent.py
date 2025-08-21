import streamlit as st
from typing import Dict
from ..core.portfolio import Portfolio
from ..core.market_data import MarketData
from ..utils.visualizer import PortfolioVisualizer
from ..workflow.state import AgentState
from ..rag.rag_system import RAGSystem
from ..agents.market_analysis_agent import MarketAnalysisAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import datetime as dt
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import yfinance as yf
from io import StringIO

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PortfolioAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=512)
        self.rag = RAGSystem([
            "fin_goal_planning_agent.json",
            "fin_market_analysis_agent.json",
            "fin_portfolio_analysis_agent.json",
            "fin_qna_agent_links.json",
            "fin_tax_education_agent.json",
        ])
        # Get S&P 100 tickers from MarketAnalysisAgent
        self.sp100_tickers = MarketAnalysisAgent().sp100_tickers

    def execute(self, state: AgentState) -> AgentState:
        query = state["query"]
        if "portfolio" not in st.session_state:
            st.session_state.portfolio = {}  # Initialize as empty dict
        if "market_data" not in st.session_state:
            st.session_state.market_data = MarketData()

        response = {}
        try:
            if st.session_state.portfolio:
                tickers = list(st.session_state.portfolio.keys())
                # Validate tickers against S&P 100 (case-insensitive)
                invalid_tickers = [ticker for ticker in tickers if ticker.upper() not in [t.upper() for t in self.sp100_tickers]]
                if invalid_tickers:
                    response = {"error": f"Invalid tickers found: {', '.join(invalid_tickers)}. Please use S&P 100 tickers."}
                else:
                    # Use default dates
                    start_date = dt.datetime(2024, 1, 1)
                    end_date = dt.datetime.today()
                    # Fetch data for each ticker individually
                    prices, ticker_prices = self._fetch_data_with_retry(tickers, start_date, end_date)
                    if prices is not None and not prices.empty:
                        # Create a StringIO object to mimic CSV file
                        portfolio_csv = StringIO()
                        portfolio_csv.write("Ticker,Quantity\n")
                        for ticker, qty in st.session_state.portfolio.items():
                            portfolio_csv.write(f"{ticker},{qty}\n")
                        portfolio_csv.seek(0)
                        # Initialize Portfolio with StringIO
                        portfolio = Portfolio(portfolio_csv)
                        st.session_state.market_data.prices = prices  # Update MarketData prices
                        analyzer = PortfolioAnalyzer(portfolio, st.session_state.market_data)
                        portfolio_returns, total_return, annualized_volatility, sharpe_ratio = analyzer.calculate_metrics()
                        if portfolio_returns is not None:
                            composition = portfolio.get_composition(prices)
                            # Calculate total price per ticker and total portfolio worth
                            total_prices = {
                                ticker: ticker_prices.get(ticker, 0) * qty 
                                for ticker, qty in zip(tickers, [st.session_state.portfolio[ticker] for ticker in tickers])
                            }
                            total_portfolio_worth = sum(total_prices.values())
                            response = {
                                "total_return": total_return,
                                "annualized_volatility": annualized_volatility,
                                "sharpe_ratio": sharpe_ratio,
                                "portfolio_returns": portfolio_returns,
                                "composition": composition,
                                "total_prices": total_prices,
                                "total_portfolio_worth": total_portfolio_worth
                            }

                            # Add LLM-generated commentary
                            context = "\n".join(self.rag.retrieve_context(query)) or "No additional context available."
                            prompt = f"""
                            You are a financial analyst. Provide a brief commentary (1-2 paragraphs) on the portfolio's performance based on the following metrics:
                            - Total Return: {total_return:.2f}%
                            - Annualized Volatility: {annualized_volatility:.2f}%
                            - Sharpe Ratio: {sharpe_ratio:.2f}
                            - Composition: {', '.join([f'{row["Ticker"]}: {row["Quantity"]} shares' for _, row in composition.iterrows()])}
                            - Total Portfolio Worth: ${total_portfolio_worth:,.2f}
                            Query: {query}
                            Context: {context}
                            """
                            commentary = self.llm.invoke([
                                SystemMessage(content="You are a knowledgeable financial analyst."),
                                HumanMessage(content=prompt)
                            ])
                            response["commentary"] = commentary.content
                        else:
                            response = {"error": "Failed to calculate portfolio metrics due to insufficient data."}
                    else:
                        response = {"error": "Failed to fetch stock data for all tickers. Please check your network or try again later."}
            else:
                response = {"error": "Please upload a CSV file with portfolio details (columns: Ticker, Quantity)."}
        except Exception as e:
            logger.error(f"Error processing portfolio: {str(e)}")
            st.error(f"Error processing portfolio: {str(e)}")
            response = {"error": f"Error processing portfolio: {str(e)}"}
        state["response"] = response
        return state

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _fetch_data_with_retry(self, tickers, start_date, end_date):
        try:
            price_data = []
            ticker_prices = {}  # Store current price for each ticker
            for ticker in tickers:
                logger.debug(f"Fetching data for ticker: {ticker}")
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(start=start_date, end=end_date)
                if hist.empty:
                    logger.error(f"No historical data returned for {ticker}")
                    raise Exception(f"No historical data for {ticker}")
                # Get current price from the latest data
                current_price = hist['Close'].iloc[-1] if not hist['Close'].empty else 0
                ticker_prices[ticker] = current_price
                hist = hist[['Close']].rename(columns={'Close': ticker})
                price_data.append(hist)
            # Combine data into a single DataFrame
            if price_data:
                prices = pd.concat(price_data, axis=1).dropna()
                return prices, ticker_prices
            else:
                raise Exception("No data retrieved for any tickers")
        except Exception as e:
            logger.error(f"Failed to fetch data for tickers {tickers}: {str(e)}")
            raise Exception(f"Failed to fetch data after retries: {str(e)}")

class PortfolioAnalyzer:
    def __init__(self, portfolio, market_data):
        self.portfolio = portfolio
        self.market_data = market_data
    
    def calculate_metrics(self):
        if self.portfolio.data is None or self.market_data.prices is None:
            return None, None, None, None
        
        returns = self.market_data.prices.pct_change().dropna()
        
        latest_prices = self.market_data.prices.iloc[-1]
        portfolio_value = sum(latest_prices[ticker] * qty for ticker, qty in zip(self.portfolio.data["Ticker"], self.portfolio.data["Quantity"]))
        weights = [(latest_prices[ticker] * qty) / portfolio_value for ticker, qty in zip(self.portfolio.data["Ticker"], self.portfolio.data["Quantity"])]
        
        portfolio_returns = (returns * weights).sum(axis=1)
        
        total_return = (self.market_data.prices.iloc[-1] / self.market_data.prices.iloc[0] - 1).mean() * 100
        annualized_volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() != 0 else 0
        
        return portfolio_returns, total_return, annualized_volatility, sharpe_ratio
