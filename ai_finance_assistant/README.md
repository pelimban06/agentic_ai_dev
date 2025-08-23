# AI Finance Assistant

A Streamlit-based application for financial tasks, including portfolio
analysis, market insights, goal planning, news synthesis, and Q&A, with
LangGraph for state management and RAG for context retrieval. Currently,
only financial news queries are routed to the `NewsSynthesizerAgent`.

## Setup

### Prerequisites
- Python 3.10
- An OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- JSON files with URLs in `data/` (e.g., `fin_qna_agent_links.json`)
- `faiss_plus_chunk.tar.gz` for FAISS indices or chunked data

### Option 1: Local Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```
   On Windows, use:
   ```bash
   set OPENAI_API_KEY=your-api-key
   ```

3. **Prepare JSON Files**:
   Place JSON files with URLs in `data/` (e.g., `fin_qna_agent_links.json`).
   Example structure:
   ```json
   [{"url": "https://www.investopedia.com/portfolio-management-4689745"}]
   ```

4. **Extract FAISS Data**:
   Place `faiss_plus_chunk.tar.gz` in your apps home directory (`./ai_finance_assistant`) and extract it:
   ```bash
   cd ~
   tar -xvzf faiss_plus_chunk.tar.gz
   ```
   This extracts FAISS indices or chunked data used by the RAG system.

5. **Run the App**:
   ```bash
   python run_app.py
   ```
   Access the app at `http://localhost:8501`.

### Option 2: Docker Setup
1. **Extract FAISS Data**:
   ```bash
   cd ./ai_finance_assistant
   tar -zxvpf src/data/faiss_plus_chunk.tar.gz --strip-components=1 && find . -maxdepth 1 -type f -exec mv {} . \;
   ```
2. **Build the Docker Image**:
   Ensure a `Dockerfile` is in the project root (see example below).
   ```bash
   docker build -t localhost/ai_finance_assistant .
   ```
   For Podman users, this command is compatible and avoids registry resolution issues.

3. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here localhost/ai_finance_assistant
   ```
   - Replace `your_key_here` with your OpenAI API key.
   - Access the app at `http://localhost:8501`.
     ```bash
     docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here -v ~/faiss_plus_chunk:/usr/src/app/faiss_data localhost/ai_finance_assistant
     ```

   **Note**: If using Podman, you may see a warning about emulating Docker CLI. Suppress it by running:
   ```bash
   sudo touch /etc/containers/nodocker
   ```

## Usage
- Enter a query (e.g., "Latest stock market news") in the Streamlit UI.
- Only financial news queries are currently supported and routed to `NewsSynthesizerAgent`.
- Non-news queries (e.g., "Analyze my portfolio") will return "No, this question cannot be answered by this app."
- Use the sidebar for portfolio CSV uploads or goal creation (if re-enabled in `agent_router.py`).

## Directory Structure
- `src/agents/`  : Agent classes (e.g., `NewsSynthesizerAgent`).
- `src/core/`    : Core data models (Portfolio, MarketData, FinancialGoal).
- `src/data/`    : JSON files for RAG (e.g., `fin_qna_agent_links.json`).
- `src/rag/`     : RAG system for context retrieval.
- `src/web_app/` : Streamlit app logic (`app.py`).
- `src/utils/`   : Visualization utilities.
- `src/workflow/`: LangGraph workflow and routing (`agent_router.py`).
- `src/config/`  : Configuration files.
- `tests/`       : Unit tests.
- `run_app.py`   : Entry point for the Streamlit app.
- `requirements.txt`: Project dependencies.
- `data/`        : Directory for JSON data files and FAISS data (e.g., extracted `faiss_plus_chunk.tar.gz`).
- 'chunked_data.json' :
- 'chunked_metadata.json' :
- 'fin_faiss_index' :

## Setup Instructions
### Local Setup
1. **Create Directory Structure**:
   Ensure directories (`src/`, `data/`, `tests/`) and files are set up as above.
   Place JSON files in `data/` with valid URLs (e.g., Investopedia links).

2. **Extract FAISS Data**:
   Extract `src/data/faiss_plus_chunk.tar.gz` in your app's home directory:
   ```bash
   cd ./ai_finance_assistant
   tar -zxvpf src/data/faiss_plus_chunk.tar.gz --strip-components=1 && find . -maxdepth 1 -type f -exec mv {} . \;
   ```

3. **Environment Variables**:
   Set `OPENAI_API_KEY`:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the App**:
   ```bash
   python run_app.py
   ```

### Docker Setup
1. **Extract FAISS Data**:
   Extract `faiss_plus_chunk.tar.gz` in your home directory as above.

2. **Create Dockerfile**:
   Save the example Dockerfile above in the project root.

3. **Build and Run**:
   Follow the Docker instructions above to build and run the container.

## How It Works
- **RAGSystem**: Loads URLs from JSON files in `data/` and FAISS indices from extracted `faiss_plus_chunk.tar.gz`, fetches content, chunks it, and creates a FAISS vector store with HuggingFaceEmbeddings for context retrieval.
- **AgentRouter**: Uses ChatOpenAI (gpt-4o-mini) with RAG context to route queries to `NewsSynthesizerAgent` for financial news; other queries are rejected.
- **Agents**: Currently, only `NewsSynthesizerAgent` is active for news-related tasks.
- **Web App**: Provides a Streamlit UI for user queries and agent outputs.
- **Workflow**: Manages state and routing with LangGraph.
- **Config**: Centralizes settings in `src/config/config.yaml`.

## Notes
- **JSON Files**: Ensure `data/` contains valid JSON files (e.g., `fin_qna_agent_links.json`). If unavailable, create sample files or modify `RAGSystem` to use static documents.
- **FAISS Data**: Extract `src/data/faiss_plus_chunk.tar.gz` to `./ai_finance_assistant` to provide FAISS indices or chunked data for `RAGSystem`. If needed in `data/`, copy the extracted files there.
- **OpenAI Dependency**: Requires an OpenAI API key. Alternatively, explore xAI’s API (see https://x.ai/api).
- **Date**: Uses today’s date (August 23, 2025). For testing, hardcode `today = dt.date(2025, 8, 23)` in `FinancialGoal.months_to_deadline`.
- **Extensibility**: To enable other agents (e.g., `PortfolioAnalysisAgent`), modify `agent_router.py` to route additional query types.
- **Docker/Podman**: Use `localhost/ai_finance_assistant` for Podman compatibility to avoid registry errors.

