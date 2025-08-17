# AI Finance Assistant

A Streamlit-based application for financial tasks, including portfolio
analysis, market insights, goal planning, news synthesis, and Q&A, with
LangGraph for state management and RAG for context retrieval.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

3. **Prepare JSON Files**:
   Place JSON files with URLs in `src/data/json_data/` (e.g., `fin_goal_planning_agent.json`).
   Example structure:
   ```json
   [{"url": "https://www.investopedia.com/portfolio-management-4689745"}]
   ```

4. **Run the App**:
   ```bash
   streamlit run src/web_app/app.py
   ```

## Usage

- Enter a query (e.g., "Analyze my portfolio", "Check AAPL stock").
- Use the sidebar for portfolio CSV uploads or goal creation.
- The app routes queries to the appropriate agent using RAG and ChatOpenAI.

## Directory Structure

- `src/agents/`  : Agent classes (Finance Q&A, Portfolio Analysis, etc.).
- `src/core/`    : Core data models (Portfolio, MarketData, FinancialGoal).
- `src/data/`    : JSON files for RAG.
- `src/rag/`     : RAG system for context retrieval.
- `src/web_app/` : Streamlit app logic.
- `src/utils/`   : Visualization utilities.
- `src/workflow/`: LangGraph workflow and routing.
- `src/config/`  : Configuration files.
- `tests/`       : Unit tests.

## Setup Instructions

1. **Create Directory Structure**:
   Create the directories and files as outlined above.
   Place JSON files in `src/data/json_data/` with valid URLs (e.g., Investopedia links).

2. **Environment Variables**:
   Set `OPENAI_API_KEY`:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**:
   ```bash
   streamlit run src/web_app/app.py
   ```

## How It Works

- **RAGSystem**: Loads URLs from JSON files, fetches content, chunks it, and
    creates a FAISS vector store with HuggingFaceEmbeddings. Retrieves context
   	for routing.

- **AgentRouter**: Uses ChatOpenAI (gpt-4o) with RAG context to route queries
    to agents.

- **Agents**: Handle specific tasks (portfolio analysis, goal planning, etc.).

- **Web App**: Provides a Streamlit UI for user queries and agent outputs.

- **Workflow**: Manages state and routing with LangGraph.

- **Config**: Centralizes settings in `config.yaml`.

## Notes

- **JSON Files**: Ensure `src/data/json_data/` contains valid JSON files with
                  URLs. If unavailable, create sample files or modify RAGSystem
				  to use static documents.

- **OpenAI Dependency**: Requires an OpenAI API key. Alternatively, use xAI’s
                         API (see https://x.ai/api for details).

- **Date**: Uses today’s date (August 16, 2025). Hardcode `today = dt.date
            (2025, 8, 16)` in `FinancialGoal.months_to_deadline` for testing.

- **Extensibility**: Add LLM integration for `FinanceQAAgent` and 
                     `TaxEducationAgent`, or expand tests in `tests/`.
