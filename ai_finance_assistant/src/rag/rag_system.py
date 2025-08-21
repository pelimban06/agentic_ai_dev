import streamlit as st
import os
import json
import re
import time
from typing import List, Dict
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, json_files: List[str], json_directory: str = "./src/data/json_data", faiss_index_path: str = "fin_faiss_index"):
        self.json_files = json_files
        self.json_directory = json_directory
        self.faiss_index_path = faiss_index_path
        try:
            self.embeddings = OpenAIEmbeddings()
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {str(e)}. Ensure OPENAI_API_KEY is set.")
            self.embeddings = None
        self.vector_store = None
        self._initialize_vector_store()

    def _extract_urls_from_json(self, file_path: str) -> List[str]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            urls = [item["url"] for item in data if "url" in item]
            return urls
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def _load_web_content(self, urls: List[str]) -> List[Dict]:
        loaded_content = []
        for url in urls:
            try:
                print(f"Loading content from {url}...")
                loader = WebBaseLoader(url)
                docs = loader.load()
                content = "\n".join([doc.page_content for doc in docs])
                loaded_content.append({"url": url, "content": content})
                print(f"Successfully loaded content from {url}")
                time.sleep(1)
            except Exception as e:
                print(f"Error loading {url}: {e}")
        return loaded_content

    def _clean_url_list(self, url_list: List) -> List[str]:
        cleaned = []
        for item in url_list:
            if isinstance(item, str):
                cleaned.append(item)
            elif isinstance(item, list):
                cleaned.extend(url for url in item if isinstance(url, str))
        return cleaned

    def _extract_urls_from_json_file(self) -> List[str]:
        url_list = []
        for json_file in self.json_files:
            file_path = os.path.join(self.json_directory, json_file)
            if not os.path.exists(file_path):
                print(f"File {file_path} not found, skipping...")
                continue
            print(f"Processing {file_path}...")
            urls = self._extract_urls_from_json(file_path)
            if not urls:
                print(f"No URLs found in {file_path}")
                continue
            url_list.extend(urls)
        return url_list

    def _generate_text_chunks(self, urls: List[str]) -> List[Dict]:
        if not urls:
            return []
        loaded_content = self._load_web_content(urls)
        content_length = 0
        count = 0
        for item in loaded_content:
            item['content'] = re.sub(r'\n\s*\n+', '\n', item['content']).strip()
            content_length += len(item['content'])
            count += 1
        print(f"Total content length: {content_length}, Number of URLs: {count}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunked_data = []
        for item in loaded_content:
            url = item["url"]
            content = item.get("content", "")
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                chunked_data.append({
                    "url": url,
                    "content": chunk,
                    "chunk_id": f"{url}_chunk_{i}"
                })
        with open("chunked_data.json", "w") as f:
            json.dump(chunked_data, f, indent=2)
        return chunked_data

    def _generate_embeddings_and_save(self, chunked_data: List[Dict]) -> FAISS:
        if not chunked_data or not self.embeddings:
            return None
        documents = [
            Document(
                page_content=item["content"],
                metadata={"chunk_id": item["chunk_id"], "url": item["url"]}
            )
            for item in chunked_data
        ]
        try:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            vector_store.save_local(self.faiss_index_path)
            with open("chunked_metadata.json", "w") as f:
                json.dump(chunked_data, f, indent=2)
            print(f"Saved FAISS vector store to {self.faiss_index_path}")
            return vector_store
        except Exception as e:
            st.error(f"Failed to create FAISS vector store: {str(e)}")
            return None

    def _initialize_vector_store(self):
        if "vector_store" not in st.session_state:
            if os.path.exists(self.faiss_index_path):
                try:
                    print(f"Loading existing FAISS vector store from {self.faiss_index_path}")
                    st.session_state.vector_store = FAISS.load_local(
                        self.faiss_index_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    print(f"Successfully loaded FAISS vector store")
                except Exception as e:
                    st.error(f"Failed to load FAISS vector store: {str(e)}")
                    st.session_state.vector_store = None
            else:
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
                os.environ["SER_AGENT"] = "investopedia_rag"
                urls = self._extract_urls_from_json_file()
                if urls:
                    chunked_data = self._generate_text_chunks(urls)
                    if chunked_data:
                        st.session_state.vector_store = self._generate_embeddings_and_save(chunked_data)
                    else:
                        st.session_state.vector_store = None
                else:
                    st.session_state.vector_store = None
        self.vector_store = st.session_state.vector_store

    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        if not self.vector_store or not self.embeddings:
            print("Debug: No vector store or embeddings available")
            return []
        try:
            #print(f"Debug: Retrieving context for query: {query}")
            docs = self.vector_store.similarity_search(query, k=k)
            context = [doc.page_content for doc in docs]
            #print(f"Debug: Retrieved documents: {[doc[:50] for doc in context]}")
            return context
        except Exception as e:
            st.error(f"RAG error: {str(e)}")
            print(f"Debug: RAG error: {str(e)}")
            return []
