import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agents.goal_planning_agent import FinancialGoal, GoalPlanner

# Fixture to create a sample FinancialGoal
@pytest.fixture
def sample_goal():
    deadline = datetime.now() + timedelta(days=365 * 5)  # 5 years from now
    return FinancialGoal("TestGoal", 100000.0, deadline, 10000.0, 5.0)

# Test GoalPlanner initialization
def test_goal_planner_init(sample_goal):
    planner = GoalPlanner(sample_goal)
    assert planner.goal == sample_goal

# Test required_monthly_savings with zero interest
def test_required_monthly_savings_zero_interest(sample_goal):
    sample_goal.interest_rate = 0.0
    planner = GoalPlanner(sample_goal)
    months = sample_goal.months_to_deadline()
    expected = (sample_goal.target_amount - sample_goal.current_savings) / months
    assert planner.required_monthly_savings() == pytest.approx(expected, rel=1e-7)

# Test required_monthly_savings with positive interest
def test_required_monthly_savings_positive_interest(sample_goal):
    planner = GoalPlanner(sample_goal)
    monthly_savings = planner.required_monthly_savings()
    assert isinstance(monthly_savings, float)
    assert monthly_savings > 0

# Test required_monthly_savings with past deadline
def test_required_monthly_savings_past_deadline():
    past_deadline = datetime.now() - timedelta(days=365)
    goal = FinancialGoal("PastGoal", 10000.0, past_deadline, 5000.0, 2.0)
    planner = GoalPlanner(goal)
    assert planner.required_monthly_savings() == 0.0

# Test project_savings
def test_project_savings(sample_goal):
    planner = GoalPlanner(sample_goal)
    monthly_savings = planner.required_monthly_savings()
    projection = planner.project_savings(monthly_savings)
    assert "projected_total" in projection
    assert "shortfall" in projection
    assert "surplus" in projection
    assert projection["projected_total"] >= sample_goal.target_amount or projection["shortfall"] == 0
