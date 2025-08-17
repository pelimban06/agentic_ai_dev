import streamlit as st
from typing import Dict, List
import datetime as dt
import math
from ..workflow.state import AgentState
from ..rag.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class FinancialGoal:
    def __init__(self, name: str, target_amount: float, deadline: dt.date, current_savings: float = 0.0, interest_rate: float = 0.0):
        self.name = name
        self.target_amount = target_amount
        self.deadline = deadline
        self.current_savings = current_savings
        self.interest_rate = interest_rate / 100
    
    def months_to_deadline(self) -> int:
        today = dt.date.today()
        months = (self.deadline.year - today.year) * 12 + (self.deadline.month - today.month)
        if self.deadline.day < today.day:
            months -= 1
        return max(months, 1)
    
    def __repr__(self):
        return f"FinancialGoal(name={self.name}, target={self.target_amount}, deadline={self.deadline})"

class GoalPlanner:
    def __init__(self, goal: FinancialGoal):
        self.goal = goal
    
    def required_monthly_savings(self) -> float:
        months = self.goal.months_to_deadline()
        if self.goal.interest_rate == 0:
            return (self.goal.target_amount - self.goal.current_savings) / months
        
        monthly_rate = self.goal.interest_rate / 12
        future_value_of_current = self.goal.current_savings * (1 + monthly_rate) ** months
        remaining = self.goal.target_amount - future_value_of_current
        if remaining <= 0:
            return 0.0
        pmt = remaining * monthly_rate / (1 - (1 + monthly_rate) ** -months)
        return pmt
    
    def project_savings(self, monthly_savings: float) -> Dict[str, float]:
        months = self.goal.months_to_deadline()
        monthly_rate = self.goal.interest_rate / 12
        projected = self.goal.current_savings
        for _ in range(months):
            projected = projected * (1 + monthly_rate) + monthly_savings
        return {
            "projected_total": projected,
            "shortfall": max(self.goal.target_amount - projected, 0),
            "surplus": max(projected - self.goal.target_amount, 0)
        }

class InnerGoalPlanningAgent:
    def __init__(self):
        self.goals: List[FinancialGoal] = []
    
    def add_goal(self, name: str, target_amount: float, deadline: dt.date, current_savings: float = 0.0, interest_rate: float = 0.0):
        goal = FinancialGoal(name, target_amount, deadline, current_savings, interest_rate)
        self.goals.append(goal)
        return goal
    
    def list_goals(self) -> List[FinancialGoal]:
        return self.goals
    
    def plan_for_goal(self, goal_name: str) -> Dict[str, any]:
        goal = next((g for g in self.goals if g.name == goal_name), None)
        if not goal:
            raise ValueError(f"Goal '{goal_name}' not found.")
        planner = GoalPlanner(goal)
        monthly_savings = planner.required_monthly_savings()
        projection = planner.project_savings(monthly_savings)
        return {
            "goal": goal,
            "required_monthly_savings": monthly_savings,
            "projection": projection
        }
    
    def overall_plan(self) -> Dict[str, float]:
        total_monthly = sum(GoalPlanner(goal).required_monthly_savings() for goal in self.goals)
        return {"total_required_monthly_savings": total_monthly}

class GoalPlanningAgent:
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
        if "inner_goal_agent" not in st.session_state:
            st.session_state.inner_goal_agent = InnerGoalPlanningAgent()

        agent = st.session_state.inner_goal_agent
        response = {"goals": [], "overall_plan": {}}
        
        with st.sidebar:
            st.subheader("Add New Goal")
            goal_name = st.text_input("Goal Name")
            target_amount = st.number_input("Target Amount ($)", min_value=0.0, step=100.0)
            deadline = st.date_input("Deadline", min_value=dt.date.today())
            current_savings = st.number_input("Current Savings ($)", min_value=0.0, step=100.0)
            interest_rate = st.number_input("Expected Annual Interest Rate (%)", min_value=0.0, max_value=20.0, step=0.1)
            
            if st.button("Add Goal"):
                if goal_name and target_amount > 0 and deadline > dt.date.today():
                    agent.add_goal(goal_name, target_amount, deadline, current_savings, interest_rate)
                    response["new_goal"] = goal_name

        try:
            goals = agent.list_goals()
            if goals:
                goal_plans = []
                for goal in goals:
                    try:
                        plan = agent.plan_for_goal(goal.name)
                        goal_plans.append({
                            "name": goal.name,
                            "target_amount": goal.target_amount,
                            "deadline": goal.deadline,
                            "current_savings": goal.current_savings,
                            "interest_rate": goal.interest_rate * 100,
                            "required_monthly_savings": plan["required_monthly_savings"],
                            "projection": plan["projection"]
                        })
                    except ValueError as e:
                        goal_plans.append({"name": goal.name, "error": str(e)})
                response["goals"] = goal_plans
                response["overall_plan"] = agent.overall_plan()

                # Add LLM-generated advice
                context = "\n".join(self.rag.retrieve_context(query))
                if not context:
                    context = "No additional context available."
                prompt = f"""
                You are a financial planner. Provide advice (1-2 paragraphs) on achieving the user's financial goals based on the query and context.
                Query: {query}
                Context: {context}
                Current Goals: {', '.join([f'{g["name"]}: ${g["target_amount"]:,.2f} by {g["deadline"]}' for g in goal_plans])}
                """
                advice = self.llm.invoke([
                    SystemMessage(content="You are a knowledgeable financial planner."),
                    HumanMessage(content=prompt)
                ])
                response["advice"] = advice.content
        except Exception as e:
            st.error(f"Error processing goals: {str(e)}")
            response = {"error": f"Error processing goals: {str(e)}"}
        state["response"] = response
        return state
