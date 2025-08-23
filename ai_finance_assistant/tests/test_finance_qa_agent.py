import unittest
from unittest.mock import Mock, patch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agents.finance_qa_agent import FinanceQAAgent
from src.workflow.state import AgentState


class TestFinanceQAAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent = FinanceQAAgent()
        self.mock_llm = Mock()
        self.state = AgentState(query="What is compound interest?")
        
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
            "Compound interest is interest calculated on the initial principal and also on the accumulated interest.",
            "It grows faster over time compared to simple interest."
        ]
        # Mock LLM response
        self.mock_llm.invoke.return_value = Mock(content="Compound interest is the process where interest is earned on both the initial principal and the interest accumulated from previous periods. This leads to exponential growth of your investment over time, unlike simple interest, which only calculates interest on the principal.")

        # Execute the agent
        result_state = self.agent.execute(self.state)

        # Verify RAG and LLM were called correctly
        self.mock_rag.retrieve_context.assert_called_once_with("What is compound interest?")
        self.mock_llm.invoke.assert_called_once()
        # Verify state update
        self.assertEqual(
            result_state["response"],
            {"answer": "Compound interest is the process where interest is earned on both the initial principal and the interest accumulated from previous periods. This leads to exponential growth of your investment over time, unlike simple interest, which only calculates interest on the principal."}
        )

    def test_execute_with_empty_context(self):
        """Test execute method when RAG returns no context."""
        # Set up RAG mock for this test
        self.agent.rag = self.mock_rag
        self.agent.llm = self.mock_llm
        
        # Mock RAG to return empty context
        self.mock_rag.retrieve_context.return_value = []
        # Mock LLM response
        self.mock_llm.invoke.return_value = Mock(content="Compound interest is when interest is added to the principal and then earns interest itself, leading to exponential growth.")

        # Execute the agent
        result_state = self.agent.execute(self.state)

        # Verify RAG and LLM were called correctly
        self.mock_rag.retrieve_context.assert_called_once_with("What is compound interest?")
        self.mock_llm.invoke.assert_called_once()
        # Verify state update
        self.assertEqual(
            result_state["response"],
            {"answer": "Compound interest is when interest is added to the principal and then earns interest itself, leading to exponential growth."}
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
        mock_streamlit_error.assert_called_once_with("Error generating finance QA response: RAG retrieval failed")
        # Verify state update
        self.assertEqual(
            result_state["response"],
            {"answer": "Sorry, an error occurred while processing your query. Please try again."}
        )

if __name__ == '__main__':
    unittest.main()
