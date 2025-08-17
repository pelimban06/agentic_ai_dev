import streamlit as st
from typing import Dict
from ..core.portfolio import Portfolio
from ..core.market_data import MarketData
from ..utils.visualizer import PortfolioVisualizer
from ..workflow.state import AgentState
from ..rag.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import datetime as dt

class PortfolioAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, max_tokens=512)
        self.rag = RAGSystem([
            "fin_goal_planning_agent.json",
            "fin_market_analysis_agent.json",
            "fin_portfolio_analysis_agent.json",
            "fin_qna_agent_links.json",
            "fin_tax_education_agent.json",
        ])

    def execute(self, state: AgentState) -> AgentState:
        query = state["query"]
        if "portfolio" not in st.session_state:
            st.session_state.portfolio = None
        if "market_data" not in st.session_state:
            st.session_state.market_data = MarketData()

        response = {}
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload CSV with Ticker and Quantity", type=["csv"])
            start_date = st.date_input("Start Date", value=dt.datetime(2024, 1, 1))
            end_date = st.date_input("End Date", value=dt.datetime.today())

        try:
            if uploaded_file is not None:
                st.session_state.portfolio = Portfolio(uploaded_file)
                tickers = st.session_state.portfolio.get_tickers()
                if tickers:
                    prices = st.session_state.market_data.fetch_data(tickers, start_date, end_date)
                    if prices is not None:
                        analyzer = PortfolioAnalyzer(st.session_state.portfolio, st.session_state.market_data)
                        portfolio_returns, total_return, annualized_volatility, sharpe_ratio = analyzer.calculate_metrics()
                        composition = st.session_state.portfolio.get_composition(prices)
                        response = {
                            "total_return": total_return,
                            "annualized_volatility": annualized_volatility,
                            "sharpe_ratio": sharpe_ratio,
                            "portfolio_returns": portfolio_returns,
                            "composition": composition
                        }

                        # Add LLM-generated commentary
                        context = "\n".join(self.rag.retrieve_context(query))
                        if not context:
                            context = "No additional context available."
                        prompt = f"""
                        You are a financial analyst. Provide a brief commentary (1-2 paragraphs) on the portfolio's performance based on the following metrics:
                        - Total Return: {total_return:.2f}%
                        - Annualized Volatility: {annualized_volatility:.2f}%
                        - Sharpe Ratio: {sharpe_ratio:.2f}
                        - Composition: {', '.join([f'{row["Ticker"]}: {row["Quantity"]} shares' for _, row in composition.iterrows()])}
                        Query: {query}
                        Context: {context}
                        """
                        commentary = self.llm.invoke([
                            SystemMessage(content="You are a knowledgeable financial analyst."),
                            HumanMessage(content=prompt)
                        ])
                        response["commentary"] = commentary.content
                    else:
                        response = {"error": "Failed to fetch stock data."}
                else:
                    response = {"error": "No valid tickers found in portfolio."}
            else:
                response = {"error": "Please upload a CSV file with portfolio details (columns: Ticker, Quantity)."}
        except Exception as e:
            st.error(f"Error processing portfolio: {str(e)}")
            response = {"error": f"Error processing portfolio: {str(e)}"}
        state["response"] = response
        return state

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
