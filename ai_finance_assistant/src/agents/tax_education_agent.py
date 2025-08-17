from typing import Dict
from ..workflow.state import AgentState
from ..rag.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st

class TaxEducationAgent:
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
            You are a tax education expert. Provide a clear and concise explanation (1-2 paragraphs) of the tax-related topic in the user's query, based on the provided context. Ensure the response is accurate and easy to understand.
            Query: {query}
            Context: {context}
            """

            # Generate response using LLM
            response = self.llm.invoke([
                SystemMessage(content="You are a knowledgeable tax education expert."),
                HumanMessage(content=prompt)
            ])
            state["response"] = {"answer": response.content}
        except Exception as e:
            st.error(f"Error generating tax education response: {str(e)}")
            state["response"] = {"error": f"Error generating tax education response: {str(e)}"}
        return state
