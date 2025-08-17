from typing import Dict
from ..workflow.state import AgentState
from ..rag.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st

class FinanceQAAgent:
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
            You are a financial education assistant. Answer the user's query based on the provided context and your knowledge.
            Query: {query}
            Context: {context}
            Provide a clear, concise, and accurate response in 1-2 paragraphs.
            """

            # Generate response using LLM
            response = self.llm.invoke([
                SystemMessage(content="You are a knowledgeable financial assistant."),
                HumanMessage(content=prompt)
            ])
            state["response"] = {"answer": response.content}
        except Exception as e:
            st.error(f"Error generating finance QA response: {str(e)}")
            state["response"] = {"answer": "Sorry, an error occurred while processing your query. Please try again."}
        return state
