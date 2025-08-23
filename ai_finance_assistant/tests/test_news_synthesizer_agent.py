import unittest
from unittest.mock import Mock, patch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agents.news_synthesizer_agent import NewsSynthesizerAgent
from src.workflow.state import AgentState


class TestNewsSynthesizerAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent = NewsSynthesizerAgent()
        self.mock_llm = Mock()
        self.state = AgentState(query="Recent stock market trends")
        
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
            "Stock markets rose due to positive earnings reports.",
            "Tech sector led gains with strong AI company performances."
        ]
        # Mock LLM response
        self.mock_llm.invoke.return_value = Mock(content="Recent stock market trends show gains driven by positive earnings, with the tech sector, particularly AI companies, leading the rally.")

        # Execute the agent
        result_state = self.agent.execute(self.state)

        # Verify RAG and LLM were called correctly
        self.mock_rag.retrieve_context.assert_called_once_with("Recent stock market trends")
        self.mock_llm.invoke.assert_called_once()
        # Verify state update
        self.assertEqual(
            result_state["response"],
            {"summary": "Recent stock market trends show gains driven by positive earnings, with the tech sector, particularly AI companies, leading the rally."}
        )

    def test_execute_with_empty_context(self):
        """Test execute method when RAG returns no context."""
        # Set up RAG mock for this test
        self.agent.rag = self.mock_rag
        self.agent.llm = self.mock_llm
        
        # Mock RAG to return empty context
        self.mock_rag.retrieve_context.return_value = []
        # Mock LLM response
        self.mock_llm.invoke.return_value = Mock(content="No specific financial news available, but markets are generally influenced by economic indicators.")

        # Execute the agent
        result_state = self.agent.execute(self.state)

        # Verify RAG and LLM were called correctly
        self.mock_rag.retrieve_context.assert_called_once_with("Recent stock market trends")
        self.mock_llm.invoke.assert_called_once()
        # Verify state update
        self.assertEqual(
            result_state["response"],
            {"summary": "No specific financial news available, but markets are generally influenced by economic indicators."}
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
        mock_streamlit_error.assert_called_once_with("Error generating news summary: RAG retrieval failed")
        # Verify state update
        self.assertEqual(
            result_state["response"],
            {"error": "Error generating news summary: RAG retrieval failed"}
        )

if __name__ == '__main__':
    unittest.main()
