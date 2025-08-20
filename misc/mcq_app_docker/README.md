# PDF Q&A and MCQ Generator

This is a Streamlit-based web application that allows users to upload a PDF file, extract its text, and either ask questions about the content or generate multiple-choice questions (MCQs) using the OpenAI Chat API.

## Features
- Upload PDF files and extract text for processing.
- Ask specific questions about the PDF content and receive answers powered by OpenAI's GPT models.
- Generate a customizable number of MCQs (with 4 options each) based on the PDF content.
- Simple and intuitive web interface built with Streamlit.

## Prerequisites
- Python 3.8 or higher
- An OpenAI API key (obtainable from [OpenAI](https://platform.openai.com/))
- A PDF file for testing


# Running the Streamlit Application with Docker (Recommended)

To run the Streamlit app using Docker, follow these steps:

## 1. Build the Docker Image
Navigate to the directory containing the `Dockerfile` and run the following command to build the Docker image:

```bash
docker build -t mcq-app .
```

## 2. Launch the Docker Container
Run the Docker image, mapping port `8501` and passing your OpenAI API key. Replace `<your-api-key>` with your actual OpenAI API key:

```bash
docker run -p 8501:8501 -e OPENAI_API_KEY="<your-api-key>" mcq-app
```

## 3. Access the Application
After running the container, you’ll see output like this in your terminal:

```
Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501
```

Open your web browser and navigate to `http://localhost:8501` to use the application. If `localhost` doesn’t work, try `http://127.0.0.1:8501`.

**Note**: If you’re using Podman instead of Docker, you may see a message about emulating Docker CLI. This can be ignored or suppressed by creating `/etc/containers/nodocker`.

## Usage
1. Open the application in your browser (typically at `http://localhost:8501`).
2. Upload a PDF file using the file uploader.
3. Choose whether to:
   - **Ask a Question**: Enter a question about the PDF content and click "Get Answer" to receive a response.
   - **Generate MCQs**: Specify the number of MCQs (1–10) and click "Generate MCQs" to view the questions.
4. Optionally, view the extracted text from the PDF for reference.

## Notes
- **PDF Limitations**: Text extraction quality depends on the PDF. Some PDFs (e.g., scanned images) may require OCR for better results. Consider using libraries like `pdfplumber` or `pymupdf` for improved extraction.
- **Token Limits**: The app limits context to approximately 2,000 tokens to stay within OpenAI's model limits (e.g., `gpt-4o-mini` has a 128,000-token context but a 4,096-token output cap).
- **Extending for DOCX**: To support DOCX files, add `type=["pdf", "docx"]` to the file uploader and install `docx2txt` for text extraction.
- **Security**: Keep your OpenAI API key secure and never commit `.streamlit/secrets.toml` to version control.
- **Model Choice**: The app uses `gpt-4o-mini` by default. You can change to other models (e.g., `gpt-4o`, `gpt-3.5-turbo`) by modifying the code, but ensure your API key has access and adjust token limits accordingly.

## Troubleshooting
- **API Key Errors**: Ensure the OpenAI API key is valid and correctly added to `.streamlit/secrets.toml`.
- **PDF Extraction Issues**: If text extraction fails, check if the PDF is text-based or image-based. For image-based PDFs, consider integrating an OCR library like `pytesseract`.
- **Token Limit Errors**: If you receive "max_tokens" or context length errors, reduce the input text size or adjust `max_context_tokens` in the code.

## License
This project is licensed under the MIT License.
