import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from datetime import datetime
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agents.market_analysis_agent import MarketAnalysisAgent
from src.workflow.state import AgentState
from src.rag.rag_system import RAGSystem

from unittest.mock import patch, MagicMock

# Fixture to create a sample AgentState
@pytest.fixture
def sample_state():
    return AgentState(query="Check AAPL stock", agent_name="", decision="market", rag_context=[], response={}, messages=[])

# Fixture to mock yfinance data
@pytest.fixture
def mock_yfinance_data(mocker):
    mock_hist = pd.DataFrame({
        'Close': [150.0, 152.0, 155.0, 160.0],
    }, index=pd.date_range(start="2024-01-01", periods=4))
    mock_info = {
        "currentPrice": 160.0,
        "marketCap": 2500000000000.0,
        "fiftyTwoWeekHigh": 170.0,
        "fiftyTwoWeekLow": 130.0,
        "trailingPE": 25.0,
        "averageVolume": 85000000.0,
        "dividendYield": 0.01
    }
    mocker.patch('yfinance.Ticker', return_value=MagicMock(info=mock_info, history=MagicMock(return_value=mock_hist)))
    return mock_hist, mock_info

# Fixture to mock RAG context
@pytest.fixture
def mock_rag_context(mocker):
    mocker.patch('src.rag.rag_system.RAGSystem', return_value=MagicMock())
    agent = MarketAnalysisAgent()
    agent.rag = MagicMock()
    agent.rag.retrieve_context.return_value = ["Market context for AAPL"]
    return agent

# Fixture to mock ChatOpenAI creation
@pytest.fixture
def mock_llm(mocker):
    mock_llm = MagicMock()
    mocker.patch('langchain_openai.ChatOpenAI', return_value=mock_llm)
    return mock_llm

# Test MarketAnalysisAgent initialization
def test_market_analysis_agent_init(mock_llm):
    with patch('src.agents.market_analysis_agent.ChatOpenAI', return_value=MagicMock()) as mock_llm:
        agent = MarketAnalysisAgent()
        assert isinstance(agent.llm, MagicMock)

# Test _fetch_stock_data success
def test_fetch_stock_data_success(mock_yfinance_data, mock_llm):
    agent = MarketAnalysisAgent()
    ticker = "AAPL"
    result = agent._fetch_stock_data(ticker)
    assert "error" not in result
    assert result["ticker"] == ticker
    assert result["current_price"] == 160.0
    assert isinstance(result["historical_data"], pd.DataFrame)
    assert not result["historical_data"].empty

# Test _fetch_stock_data failure
def test_fetch_stock_data_failure(mock_yfinance_data, mock_llm, mocker):
    agent = MarketAnalysisAgent()
    mocker.patch.object(agent, '_fetch_stock_data', side_effect=lambda ticker: {"error": f"Failed to fetch data for {ticker}: API error"})
    result = agent._fetch_stock_data("AAPL")
    assert "error" in result
    assert "Failed to fetch data for AAPL" in result["error"]

# Test _generate_analysis success
def test_generate_analysis_success(mock_yfinance_data, mock_rag_context, mock_llm):
    agent = mock_rag_context
    ticker = "AAPL"
    data = agent._fetch_stock_data(ticker)
    query = "Check AAPL stock"
    context = "\n".join(agent.rag.retrieve_context())
    analysis = agent._generate_analysis(ticker, data, query, context)
    assert isinstance(analysis, str)
    assert "Error generating analysis" not in analysis

# Test _generate_analysis failure
def test_generate_analysis_failure(mock_llm, mocker):
    agent = MarketAnalysisAgent()
    with patch.object(agent, 'llm', new=MagicMock()) as mock_llm:  # Ensure llm is mocked within this context
        mock_llm.invoke.side_effect = Exception("LLM error")
        print(f"Mocked llm invoke: {mock_llm.invoke}")  # Debug print
        data = {
            "ticker": "AAPL",
            "current_price": 160.0,
            "market_cap": 2500000000000.0,
            "fiftyTwoWeekHigh": 170.0,
            "fiftyTwoWeekLow": 130.0,
            "trailingPE": 25.0,
            "averageVolume": 85000000.0,
            "dividendYield": 1.0
        }
        analysis = agent._generate_analysis("AAPL", data, "Check AAPL stock", "Context")
    assert "Error generating analysis" in analysis

# Test execute with valid ticker
def test_execute_valid_ticker(mock_yfinance_data, mock_rag_context, mock_llm, sample_state):
    agent = mock_rag_context
    updated_state = agent.execute(sample_state)
    response = updated_state["response"]
    assert "error" not in response
    assert "ticker" in response
    assert response["ticker"] == "AAPL"
    assert "analysis" in response
    assert isinstance(response["analysis"], str)

# Test execute with invalid ticker
def test_execute_invalid_ticker(sample_state):
    state = AgentState(query="Check XYZ stock", agent_name="", decision="market", rag_context=[], response={}, messages=[])
    agent = MarketAnalysisAgent()
    updated_state = agent.execute(state)
    response = updated_state["response"]
    assert "error" in response
    assert "Please include a valid S&P 100 ticker symbol" in response["error"]

# Test execute with exception
def test_execute_with_exception(sample_state, mocker):
    mocker.patch.object(MarketAnalysisAgent, '_fetch_stock_data', side_effect=Exception("Unexpected error"))
    agent = MarketAnalysisAgent()
    updated_state = agent.execute(sample_state)
    response = updated_state["response"]
    assert "error" in response
    assert "Error processing market data" in response["error"]
    assert "Unexpected error" in response["error"]
