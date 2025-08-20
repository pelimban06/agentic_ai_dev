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

## Setup Instructions
Follow these steps to set up and run the application locally:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv mcqvenv
   source mcqvenv/bin/activate
   ```

2. **Install dependencies**:
   Ensure you have a `requirements.txt` file (see below for contents) and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create Streamlit configuration directory**:
   ```bash
   mkdir -p .streamlit
   ```

4. **Create secrets file**:
   ```bash
   touch .streamlit/secrets.toml
   ```

5. **Add OpenAI API key**:
   Open `.streamlit/secrets.toml` in a text editor and add the following line, replacing `<your own openai key>` with your actual OpenAI API key:
   ```
   OPENAI_API_KEY="<your own openai key>"
   ```

6. **Run the application**:
   Start the Streamlit app with:
   ```bash
   streamlit run gen_mcq.py
   ```
   This will launch the application and open a browser window with the GUI, which is self-explanatory.

## requirements.txt
Create a `requirements.txt` file with the following content: or use the one in the app directory.
```
streamlit==1.39.0
PyPDF2==3.0.1
openai==1.50.2
```

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
