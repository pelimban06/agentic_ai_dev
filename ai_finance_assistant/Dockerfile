# Stage 1: Build dependencies
# Use official Python slim image for a smaller footprint
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY run_app.py .
COPY requirements.txt .
COPY chunked_data.json .
COPY chunked_metadata.json .
COPY fin_faiss_index ./fin_faiss_index
COPY src ./src
#COPY .streamlit/ .streamlit/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "run_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
