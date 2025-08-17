import streamlit as st
from typing import Dict
from ..workflow.state import AgentState
from ..rag.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class NewsSynthesizerAgent:
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
        try:
            # Retrieve context from RAG
            context = "\n".join(self.rag.retrieve_context(query))
            if not context:
                context = "No additional context available."

            # Define prompt for LLM
            prompt = f"""
            You are a financial news analyst. Provide a concise summary (1-2 paragraphs) of recent financial news related to the user's query, based on the provided context. Focus on key trends, events, or insights relevant to the query.
            Query: {query}
            Context: {context}
            """

            # Generate response using LLM
            response = self.llm.invoke([
                SystemMessage(content="You are a knowledgeable financial news analyst."),
                HumanMessage(content=prompt)
            ])
            state["response"] = {"summary": response.content}
        except Exception as e:
            st.error(f"Error generating news summary: {str(e)}")
            state["response"] = {"error": f"Error generating news summary: {str(e)}"}
        return state
