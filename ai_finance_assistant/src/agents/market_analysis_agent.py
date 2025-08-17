import streamlit as st
from typing import Dict
import yfinance as yf
from ..workflow.state import AgentState
from ..rag.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class MarketAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, max_tokens=512)
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

    def execute(self, state: AgentState) -> AgentState:
        query = state["query"]
        ticker = None
        words = query.upper().split()
        print(words)
        for word in words:
            if word in self.sp100_tickers:
                ticker = word
                break
        
        response = {}
        try:
            if not ticker:
                response = {"error": "Please include a valid S&P 100 ticker symbol in your query (e.g., AAPL, MSFT)."}
            else:
                data = yf.Ticker(ticker)
                info = data.info
                hist = data.history(period="1y")
                response = {
                    "ticker": ticker,
                    "current_price": info.get("currentPrice", "N/A"),
                    "market_cap": info.get("marketCap", "N/A"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                    "historical_data": hist
                }

                # Add LLM-generated analysis
                context = "\n".join(self.rag.retrieve_context(query))
                if not context:
                    context = "No additional context available."
                prompt = f"""
                You are a market analyst. Provide a brief analysis (1-2 paragraphs) of the stock based on the following data:
                - Ticker: {ticker}
                - Current Price: ${response['current_price']}
                - Market Cap: ${response['market_cap']:,}
                - 52-Week High: ${response['fifty_two_week_high']}
                - 52-Week Low: ${response['fifty_two_week_low']}
                Query: {query}
                Context: {context}
                """
                analysis = self.llm.invoke([
                    SystemMessage(content="You are a knowledgeable market analyst."),
                    HumanMessage(content=prompt)
                ])
                response["analysis"] = analysis.content
        except Exception as e:
            st.error(f"Error processing market data: {str(e)}")
            response = {"error": f"Error processing market data: {str(e)}"}
        state["response"] = response
        return state
