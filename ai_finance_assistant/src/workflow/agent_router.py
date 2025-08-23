from typing import Dict, List
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from pydantic.json_schema import JsonSchemaValue  # For json_schema_extra
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from .state import AgentState
import streamlit as st
from ..rag.rag_system import RAGSystem

class Route(BaseModel):
    step: str = Field(
        description="The next step in the routing process",
        json_schema_extra={"enum": ["finance", "portfolio", "market", "goal", "news", "tax"]}
    )

class AgentRouter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=512
        )
        self.router = self.llm.with_structured_output(Route)
        self.rag = RAGSystem([
            "fin_goal_planning_agent.json",
            "fin_market_analysis_agent.json",
            "fin_portfolio_analysis_agent.json",
            "fin_qna_agent_links.json",
            "fin_tax_education_agent.json",
        ])
    
    def get_agents(self):
        from ..agents import FinanceQAAgent, PortfolioAnalysisAgent, MarketAnalysisAgent, GoalPlanningAgent, NewsSynthesizerAgent, TaxEducationAgent
        return {
            "finance": FinanceQAAgent(),
            "portfolio": PortfolioAnalysisAgent(),
            "market": MarketAnalysisAgent(),
            "goal": GoalPlanningAgent(),
            "news": NewsSynthesizerAgent(),
            "tax": TaxEducationAgent(),
        }

    def route(self, state: AgentState) -> AgentState:
        router_system_prompt = """
        You are a routing assistant for a financial agents app. Based on the user query and the provided context, determine which agent to route to:
        - "finance": General financial education questions (e.g., compound interest, budgeting).
        - "portfolio": Portfolio analysis requests (e.g., analyze my portfolio, investments).
        - "market": Market insights or stock data (e.g., check AAPL, market trends).
        - "goal": Financial goal planning (e.g., plan for retirement, savings).
        - "news": Financial news requests (e.g., latest stock news).
        - "tax": Tax-related questions (e.g., Roth IRA, tax deductions).
        If the query does not fit any of these categories, respond with "none".
        Context: {context}
        Return a structured response with the 'step' field set to the appropriate agent or "none".
        """

        context = "\n".join(self.rag.retrieve_context(state["query"]))
        state["rag_context"] = self.rag.retrieve_context(state["query"])

        try:
            decision = self.router.invoke([
                SystemMessage(content=router_system_prompt.format(context=context)),
                HumanMessage(content=state["query"])
            ])
            if decision.step == "none":
                state["response"] = {"error": "I am a finance agent, this question cannot be answered by this app."}
                state["decision"] = None
                state["agent_name"] = None
            else:
                state["decision"] = decision.step
                state["agent_name"] = decision.step.capitalize()
        except Exception as e:
            st.error(f"Routing error: {str(e)}")
            state["response"] = {"error": "I am a finance agent, this question cannot be answered by this app."}
            state["decision"] = None
            state["agent_name"] = None
        return state

    def dispatch(self, state: AgentState) -> AgentState:
        agents = self.get_agents()
        agent_name = state["decision"]
        agent = agents.get(agent_name)
        if agent:
            state = agent.execute(state)
        else:
            state["response"] = {"error": "Invalid agent selected."}
        return state

def create_workflow():
    workflow = StateGraph(AgentState)
    router = AgentRouter()
    workflow.add_node("router", router.route)
    workflow.add_node("dispatch", router.dispatch)
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "dispatch")
    workflow.add_edge("dispatch", END)
    return workflow.compile()
