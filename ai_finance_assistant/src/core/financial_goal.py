import datetime as dt
import math
from typing import Dict

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
