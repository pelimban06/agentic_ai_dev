import streamlit as st
from typing import Dict
from ..workflow.state import AgentState
from ..rag.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from gnews import GNews

class NewsSynthesizerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=1500)
        self.rag = RAGSystem([
            "fin_goal_planning_agent.json",
            "fin_market_analysis_agent.json",
            "fin_portfolio_analysis_agent.json",
            "fin_qna_agent_links.json",
            "fin_tax_education_agent.json",
        ])
        self.google_news = GNews(max_results=5)  # Limit to 5 recent articles

    def execute(self, state: AgentState) -> AgentState:
        query = state["query"]
        try:
            # Retrieve context from RAG
            rag_context = "\n".join(self.rag.retrieve_context(query))
            if not rag_context:
                rag_context = "No additional context available."

            # Retrieve latest news using Google News
            news_articles = self.google_news.get_news(f"{query} financial news")
            latest_news = "\n".join([f"{article['title']}: {article['description']} (Source: {article['publisher']['title']}, URL: {article['url']})" for article in news_articles])
            if not latest_news:
                latest_news = "No latest news retrieved from Google News."

            # Combine contexts
            context = f"RAG Context: {rag_context}\nGoogle News: {latest_news}"

            # Define enhanced prompt for LLM
            prompt = f"""
            You are a financial news analyst. Based on the user's query and provided context, provide a detailed analysis including:
            - **Summary and Context**: Summarize recent financial news related to the query in 1-2 paragraphs, and contextualize it by connecting to broader industry trends, economic conditions, or relevant market dynamics.
            - **Highlights**: Identify key positive developments or opportunities.
            - **Problematic News or Risks**: Highlight any risks, challenges, or negative developments.
            - **Stock Market Impact**: Discuss news affecting the stock market, including potential impacts on specific stocks, sectors, or indices.
            - **Safe Investment Advice**: Provide safe investment strategies for volatile situations, focusing on risk mitigation (e.g., diversification, bonds, or stable assets).

            Ensure the response is structured with clear headings for each section. Use the provided context and Google News results to inform your analysis, prioritizing accuracy and clarity.
            Query: {query}
            Context: {context}
            """

            # Generate response using LLM
            response = self.llm.invoke([
                SystemMessage(content="You are a knowledgeable financial news analyst."),
                HumanMessage(content=prompt)
            ])
            state["response"] = {"analysis": response.content}
        except Exception as e:
            st.error(f"Error generating news analysis: {str(e)}")
            state["response"] = {"error": f"Error generating news analysis: {str(e)}"}
        return state
