import unittest
from unittest.mock import Mock, patch
from typing import Dict
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agents.tax_education_agent import TaxEducationAgent
from src.workflow.state import AgentState

class RAGSystem:
    def __init__(self):
        self._data_files = [
            "fin_goal_planning_agent.json",
            "fin_market_analysis_agent.json",
            "fin_portfolio_analysis_agent.json",
            "fin_qna_agent_links.json",
            "fin_tax_education_agent.json",
        ]

class TestTaxEducationAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Initialize the agent without mocking RAGSystem for initialization test
        self.agent = TaxEducationAgent()
        self.mock_llm = Mock()
        self.state = AgentState(query="What are tax deductions?")
        
        # For tests needing RAG mocking, we'll set it up individually
        self.mock_rag = Mock()

    def test_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertIsNotNone(self.agent.llm)
        self.assertIsNotNone(self.agent.rag)
        self.assertEqual(self.agent.rag._data_files, [
            "fin_goal_planning_agent.json",
            "fin_market_analysis_agent.json",
            "fin_portfolio_analysis_agent.json",
            "fin_qna_agent_links.json",
            "fin_tax_education_agent.json",
        ])

    def test_execute_with_valid_context(self):
        """Test execute method with valid context and LLM response."""
        # Set up RAG mock for this test
        self.agent.rag = self.mock_rag
        self.agent.llm = self.mock_llm
        
        # Mock RAG to return context
        self.mock_rag.retrieve_context.return_value = [
            "Tax deductions reduce taxable income.",
            "Common deductions include mortgage interest and charitable contributions."
        ]
        # Mock LLM response
        self.mock_llm.invoke.return_value = Mock(content="Tax deductions lower your taxable income, reducing the amount of tax you owe. Examples include mortgage interest and charitable contributions.")

        # Execute the agent
        result_state = self.agent.execute(self.state)

        # Verify RAG and LLM were called correctly
        self.mock_rag.retrieve_context.assert_called_once_with("What are tax deductions?")
        self.mock_llm.invoke.assert_called_once()
        # Verify state update
        self.assertEqual(
            result_state["response"],
            {"answer": "Tax deductions lower your taxable income, reducing the amount of tax you owe. Examples include mortgage interest and charitable contributions."}
        )

    def test_execute_with_empty_context(self):
        """Test execute method when RAG returns no context."""
        # Set up RAG mock for this test
        self.agent.rag = self.mock_rag
        self.agent.llm = self.mock_llm
        
        # Mock RAG to return empty context
        self.mock_rag.retrieve_context.return_value = []
        # Mock LLM response
        self.mock_llm.invoke.return_value = Mock(content="No specific information available, but tax deductions generally reduce taxable income.")

        # Execute the agent
        result_state = self.agent.execute(self.state)

        # Verify RAG and LLM were called correctly
        self.mock_rag.retrieve_context.assert_called_once_with("What are tax deductions?")
        self.mock_llm.invoke.assert_called_once()
        # Verify state update
        self.assertEqual(
            result_state["response"],
            {"answer": "No specific information available, but tax deductions generally reduce taxable income."}
        )

    @patch('streamlit.error')
    def test_execute_with_exception(self, mock_streamlit_error):
        """Test execute method when an exception occurs."""
        # Set up RAG mock for this test
        self.agent.rag = self.mock_rag
        self.agent.llm = self.mock_llm
        
        # Mock RAG to raise an exception
        self.mock_rag.retrieve_context.side_effect = Exception("RAG retrieval failed")

        # Execute the agent
        result_state = self.agent.execute(self.state)

        # Verify streamlit error was called
        mock_streamlit_error.assert_called_once_with("Error generating tax education response: RAG retrieval failed")
        # Verify state update
        self.assertEqual(
            result_state["response"],
            {"error": "Error generating tax education response: RAG retrieval failed"}
        )

if __name__ == '__main__':
    unittest.main()
