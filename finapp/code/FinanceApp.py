#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install langchain langchain-openai
#!pip install langgraph


# In[2]:


import os
import getpass
import time

from typing_extensions import Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from typing import TypedDict, List, Dict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# In[3]:


if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# In[4]:


#from langchain_openai import ChatOpenAI
# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model_name="gpt-4o",  # Specify the model (e.g., gpt-4, gpt-3.5-turbo)
    temperature=0.7,      # Controls randomness (0 to 1)
    max_tokens=512        # Maximum tokens in the response
)


# In[5]:


# AgentState
class AgentState(TypedDict):
    query_history: Annotated[List[BaseMessage], add_messages]
    system_prompt: str              # System prompt
    user_input: str
    query_type: str
    decision: str
    output: str


# In[6]:


# Schema for strctured output to use as routing logic
class Route(BaseModel):
	step: Literal["finance", "portfolio ", "market", "goal", "news", "tax"] = Field(
		None, description="The next step in the routing process"
	)

# Agument the LLM with schema for structured output
router = llm.with_structured_output(Route)


# In[7]:


def debug_print(message, level="INFO"):
    """Print debug messages with timestamp and level."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


# In[8]:


def summarize_query_history(history):
    """Summarize query history using ChatOpenAI."""
    if not history:
        return "No query history available."
    
    query_text = "\n".join(history)
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following queries in approximately 100 words:\n\n{queries}"
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=400)
    try:
        summary = llm.invoke(prompt.format(queries=query_text)).content
        return summary
    except Exception as e:
        return f"Error summarizing history: {e}"


# In[9]:


router_system_prompt = """
You are a routing assistant for a financial query system. Your task is to analyze the user's query and determine which of the following agents should handle it:

1. Finance Q&A Agent: Handles general financial education queries (e.g., explaining financial concepts, budgeting, or basic investing terms).
2. Portfolio Analysis Agent: Reviews and analyzes user portfolios (e.g., questions about portfolio performance, asset allocation, or investment strategies).
3. Market Analysis Agent: Provides real-time market insights (e.g., stock market trends, economic indicators, or market predictions).
4. Goal Planning Agent: Assists with financial goal setting and planning (e.g., retirement planning, saving for a house, or long-term financial strategies).
5. News Synthesizer Agent: Summarizes and contextualizes financial news (e.g., interpreting market news or summarizing recent financial events).
6. Tax Education Agent: Explains tax concepts and account types (e.g., tax deductions, IRAs, 401(k)s, or tax strategies).

Based on the user's query, identify the most appropriate agent to handle it. Respond with only the name of the selected agent (e.g., "finance", "portfolio", "market", "goal", "news",  "tax" , etc.). If the query is ambiguous or fits multiple agents, choose the most relevant one based on the primary focus of the query.
"""


# In[10]:


# Nodes
def call_finance_qa_agent(state: AgentState):
    """ Finance Q&A Agent """
    
    debug_print("Finance QnA Agent: Handles general financial education queries")
    result = llm.invoke(state["user_input"])
    return {"output": result.content}

def call_portfolio_analysis_agent(state: AgentState):
	""" Portfolio Analysis Agent """

	print(" Reviews and analyzes user portfolios ")
	result = llm.invoke(state["user_input"])
	return {"output": result.content}

def call_market_analysis_agent(state: AgentState):
	""" Market Analysis Agent"""

	print(" Provides real-time market insights ")
	result = llm.invoke(state["user_input"])
	return {"output": result.content}

def call_goal_planning_agent(state: AgentState):
	""" Goal Planning Agent"""

	print(" Assists with financial goal setting and planning ")
	result = llm.invoke(state["user_input"])
	return {"output": result.content}

def call_news_synthesizer_agent(state: AgentState):
	""" News Synthesizer Agent"""

	print(" Summarizes and contextualizes financial news ")
	result = llm.invoke(state["user_input"])
	return {"output": result.content}

def call_tax_education_agent(state: AgentState):
	""" Tax Education Agent"""

	print(" Explains tax concepts and account types ")
	result = llm.invoke(state["user_input"])
	return {"output": result.content}


def finapp_router(state: AgentState):
  """Route the user_input to the appropriate node """

  # Run the agumented LLM with strctured output to serve as routing logic
  decision = router.invoke(
		[
			SystemMessage(content=router_system_prompt), HumanMessage(content=state["user_input"]),
		]
	)
  print(f"decision: {decision.step}")  
  return {"decision": decision.step}


# Conditional edge function to route the appropriate node
def route_decision(state: AgentState):
	decision_str = state["decision"].rstrip()
	decision_str = decision_str.lower()
	print(decision_str)
	if decision_str is None:
		return "call_finance_qa_agent"
	
	if decision_str == 'finance':
		return "call_finance_qa_agent"
	elif decision_str == 'portfolio':
		return "call_portfolio_analysis_agent"
	elif decision_str == "market":
		return "call_market_analysis_agent"
	elif decision_str == "goal":
		return "call_goal_planning_agent"
	elif decision_str == "news":
		return "call_news_synthesizer_agent"
	elif decision_str == "tax":
		return "call_tax_education_agent"


# In[11]:


# Build workflow
router_builder = StateGraph(AgentState)

# Add nodes
router_builder.add_node("call_finance_qa_agent", call_finance_qa_agent)
router_builder.add_node("call_goal_planning_agent", call_goal_planning_agent)
router_builder.add_node("call_market_analysis_agent", call_market_analysis_agent)
router_builder.add_node("call_news_synthesizer_agent", call_news_synthesizer_agent)
router_builder.add_node("call_portfolio_analysis_agent", call_portfolio_analysis_agent)
router_builder.add_node("call_tax_education_agent", call_tax_education_agent)

router_builder.add_node("finapp_router", finapp_router)

# Add edges to connect nodes
router_builder.add_edge(START, "finapp_router")
router_builder.add_conditional_edges(
    "finapp_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "call_finance_qa_agent": "call_finance_qa_agent",
        "call_goal_planning_agent": "call_goal_planning_agent",
        "call_market_analysis_agent": "call_market_analysis_agent",
        "call_news_synthesizer_agent": "call_news_synthesizer_agent",
        "call_portfolio_analysis_agent": "call_portfolio_analysis_agent",
        "call_tax_education_agent": "call_tax_education_agent",
    },
)

router_builder.add_edge("call_finance_qa_agent", END)
router_builder.add_edge("call_goal_planning_agent", END)
router_builder.add_edge("call_market_analysis_agent", END)
router_builder.add_edge("call_news_synthesizer_agent", END)
router_builder.add_edge("call_portfolio_analysis_agent", END)
router_builder.add_edge("call_tax_education_agent", END)

# compile workflow
router_workflow = router_builder.compile()

# Show the workflow
display(Image(router_workflow.get_graph().draw_mermaid_png()))


# In[12]:


state = router_workflow.invoke({"user_input": "How can I create a budget to save for a down payment on a house?"})
print(state["output"])
#state = router_workflow.invoke({"user_input": "what is goal planning?"})
#print(state["output"])
#state = router_workflow.invoke({"input": "what is market analysis?"})
#print(state["output"])
#state = router_workflow.invoke({"user_input": "what is the news?"})
#print(state["output"])
#state = router_workflow.invoke({"user_input": "Can you provide tax education?"})
#print(state["output"])


# #### Adding RAG component

# In[26]:


import os
import time
import json
import re
import textwrap
from pprint import pprint


from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings

#from sentence_transformers import SentenceTransformer
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_community.docstore import InMemoryDocstore
from langchain_community.docstore.document import Document
#from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import openai

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# To turn debug set is_debug_on = True. Fine grained debug is not in place yet.
is_debug_on = False


# In[27]:


# Directory containing JSON files (update with your directory path)
json_directory = "./json_data"  # Replace with the path to your JSON files

# List of JSON files to process (update with your file names)
json_files = [
	"fin_goal_planning_agent.json",
	"fin_market_analysis_agent.json",
	"fin_portfolio_analysis_agent.json",
	"fin_qna_agent_links.json",
	"fin_tax_education_agent.json", 
]


# In[28]:


# Function to extract URLs from a JSON file
def extract_urls_from_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Extract URLs from the 'url' key in each dictionary
        urls = [item["url"] for item in data if "url" in item]
        return urls
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def extract_urls_from_json_file(json_files):
    url_list = []
    # Iterate through each JSON file
    for json_file in json_files:
        file_path = os.path.join(json_directory, json_file)
        if not os.path.exists(file_path):
            print(f"File {file_path} not found, skipping...")
            continue
        print(f"Processing {file_path}...")
        # Extract URLs from the JSON file
        urls = extract_urls_from_json(file_path)
        if not urls:
            print(f"No URLs found in {file_path}")
            continue
        url_list.append(urls)
#        print(urls)
#       url_counts += len(urls)
#        print("\n")

    return url_list

def clean_url_list(url_list):
    cleaned = []
    for item in url_list:
        if isinstance(item, str):
            cleaned.append(item)
        elif isinstance(item, list):
            cleaned.extend(url for url in item if isinstance(url, str))
        # Add more conditions if needed (e.g., for dictionaries)
    return cleaned


# In[29]:


def check_file(file_path):
    # Check if file exists
    if os.path.exists(file_path):
        print(f"File '{file_path}' exists.")
        # Check if file has content (size > 0)
        if os.path.getsize(file_path) > 0:
            print(f"File has content (size: {os.path.getsize(file_path)} bytes).")
            # Optionally, read and display content
            return True
        else:
            return False
    else:
        debug_print(f"File does not exist {file_path}")
        return False


# In[30]:


# Function to load web content using WebBaseLoader
def load_web_content(urls):
    loaded_content = []
    for url in urls:
        try:
            print(f"Loading content from {url}...")
            # Initialize WebBaseLoader for the URL
            loader = WebBaseLoader(url)
            # Load the content
            docs = loader.load()
            # Extract text content from the loaded documents
            content = "\n".join([doc.page_content for doc in docs])
            loaded_content.append({"url": url, "content": content})
            print(f"Successfully loaded content from {url}")
            # Add a delay to avoid overwhelming the server
            time.sleep(1)
        except Exception as e:
            print(f"Error loading {url}: {e}")
    return loaded_content


# In[32]:


def generate_text_chunks(urls):
    if not urls:
        return None

    loaded_content = []
    loaded_contents = load_web_content(urls) 

    content_length = 0
    count = 0
    for item in loaded_contents:
        item['content'] = re.sub(r'\n\s*\n+', '\n', item['content']).strip()
        content_length += len(item['content'])
        count += 1 

    print(content_length, count)
    # Step 1: Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    # Step 2: Split content and create new list of dictionaries
    chunked_data = []

    for item in loaded_contents:
        url = item["url"]
        content = item.get("content", "")  # Safely get content, default to empty string
        # Split content into chunks
        chunks = text_splitter.split_text(content)
    
        # Create a dictionary for each chunk
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "url": url,
                "content": chunk,
                "chunk_id": f"{url}_chunk_{i}"  # Unique ID for each chunk
            })

    
    # Step 3: Print or process the chunked data
    if is_debug_on:
        for chunk in chunked_data:
            print(f"URL: {chunk['url']}, Chunk ID: {chunk['chunk_id']}, Content Length: {len(chunk['content'])}")
            #Print first 50 characters of each chunk for brevity
            print(f"Content Preview: {chunk['content'][:50]}...\n")

    # Step 4 (Optional): Save chunked data to JSON
    with open("chunked_data.json", "w") as f:
        json.dump(chunked_data, f, indent=2)
        return "chunked_data.json" 

    return None




# In[37]:


def generate_embeddings_and_save(chunk_data_file):
    print("Inside generate_embeddings_and_save {generate_embeddings_and_save}")
    fin_vector_store = None
    # Step 1: Set environment variables to avoid errors
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
    os.environ["SER_AGENT"] = "investopedia_rag"  # Set SER_AGENT

    # Step 2: Load metadata and prepare documents
    try:
        with open("./chunked_data.json", "r") as f:
            metadatas = json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        exit(1)

    # Convert metadata to LangChain Documents
    documents = [
        Document(
            page_content=item["content"],
            metadata={"chunk_id": item["chunk_id"], "url": item["url"]}
        )
        for item in metadatas
    ]


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    # Step 4: Create FAISS index
    fin_vector_store = FAISS.from_documents(documents, embeddings)

    # Step 5: Save FAISS index and metadata
    print("saving fin_faiss_index")
    fin_vector_store.save_local("fin_faiss_index")  # Saves faiss_index.bin and index.pkl
    with open("chunked_metadata.json", "w") as f:
        json.dump(metadatas, f, indent=2)

    return fin_vector_store 


# In[38]:


def extract_urls_from_json_file(json_files):
    url_list = []
    # Iterate through each JSON file
    for json_file in json_files:
        file_path = os.path.join(json_directory, json_file)
        if not os.path.exists(file_path):
            print(f"File {file_path} not found, skipping...")
            continue
        print(f"Processing {file_path}...")
        # Extract URLs from the JSON file
        urls = extract_urls_from_json(file_path)
        if not urls:
            print(f"No URLs found in {file_path}")
            continue
        url_list.append(urls)
    return url_list


# In[39]:


def create_qa_chain(metadata_jfile):
  
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    # Load existing FAISS index
    vector_store = FAISS.load_local("fin_faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=400)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Combines retrieved documents into prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain	


# ### Extract data from Investopedia links and store it in a FAISS vector database for Retrieval-Augmented Generation (RAG) use.

# In[40]:


def rag_setup(chunk_json_file):
    url_list = []
    url_set = set()
    combined_list = []

    is_exist_chunk_data_file = False 
    chunk_data_file = "chunked_data.json"
    if os.path.exists(chunk_data_file):
        print(f"{chunk_data_file} is a file and exists")
        is_exist_chunk_data_file = True

    if json_files and is_exist_chunk_data_file == False:
        url_list = extract_urls_from_json_file(json_files)
        #print(url_counts)
        if url_list:
            combined_list = clean_url_list(url_list)

    # Remove duplicates
	
    if combined_list:
        url_set = set(combined_list)
        print(url_set)
        print(len(url_set))

        # Generate chunks
        url_list = []	
        url_list = list(url_set)
        chunk_data_file = generate_text_chunks(url_list);
    
    print(f"chunkedfile {chunk_data_file}")	
    if chunk_data_file:
        fin_faiss_index = generate_embeddings_and_save("chunked_data.json")
        qa_chain = create_qa_chain("chunked_metadata.json")
        
    return qa_chain


# In[41]:


query = "What is the goal settings?"
qa_chain = rag_setup("chunked_data.json")
if qa_chain:
    result = qa_chain.invoke({"query": query})
    print("Query:", query)
    print("\nGenerated Answer:", result["result"])
    print("\nSource Documents:")
    for doc in result["source_documents"]:
        print(f"Chunk ID: {doc.metadata['chunk_id']}, URL: {doc.metadata['url']}")
        print(f"Content: {doc.page_content[:]}...\n")


# In[ ]:




