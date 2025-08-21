import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import httpx

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    st.stop()

try:
    client = OpenAI(
        api_key=api_key,
        http_client=httpx.Client(proxies=None)  # Explicitly disable proxies
    )
    st.success("OpenAI client initialized successfully!")
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.stop()

st.title("PDF Q&A and MCQ Generator")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# Initialize session state for extracted text
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# Extract text from PDF
if uploaded_file:
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        st.session_state.extracted_text = text
        st.success("PDF uploaded and text extracted successfully!")
    except Exception as e:
        st.error(f"Error extracting text: {e}")

# Display extracted text
if st.checkbox("Show extracted text"):
    st.text_area("Extracted Text", st.session_state.extracted_text, height=300)

# User choice: Q&A or MCQ
action = st.radio("What would you like to do?", ("Ask a Question", "Generate MCQs"))

max_context_tokens = 2000
context = st.session_state.extracted_text[:max_context_tokens * 4]

if action == "Ask a Question":
    # Text input for user's question
    question = st.text_input("Enter your question about the PDF content:")
    
    if st.button("Get Answer") and question and context:
        try:
            # Prepare prompt for OpenAI
            prompt = f"Based on the following PDF content, answer the question: {question}\n\nContent: {context}"
            
            # Call OpenAI Chat API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            # Display the answer
            st.subheader("Answer:")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error generating answer: {e}")

elif action == "Generate MCQs":
    # Input for number of MCQs
    num_mcqs = st.number_input("Number of MCQs to generate", min_value=1, max_value=10, value=3)
    
    if st.button("Generate MCQs") and context:
        try:
            # Prepare prompt for OpenAI to generate MCQs
            prompt = f"Based on the following PDF content, generate {num_mcqs} multiple-choice questions (MCQs) with 4 options each, including the correct answer.\n\nContent: {context}"
            
            # Call OpenAI Chat API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Display the MCQs
            st.subheader("Generated MCQs:")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error generating MCQs: {e}")

# Notes
st.info("Note: This app uses OpenAI API for Q&A and MCQ generation. Ensure you have an API key set. For DOCX support, you can add 'docx' to file_uploader and use docx2txt library to extract text.")
