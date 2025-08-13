#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install langchain langchain-openai
#!pip install langgraph


# In[44]:


import os
import getpass

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


# In[37]:


if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# In[38]:


#from langchain_openai import ChatOpenAI
# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model_name="gpt-4o",  # Specify the model (e.g., gpt-4, gpt-3.5-turbo)
    temperature=0.7,      # Controls randomness (0 to 1)
    max_tokens=512        # Maximum tokens in the response
)


# In[39]:


# AgentState
class AgentState(TypedDict):
    query_history: Annotated[List[BaseMessage], add_messages]
    system_prompt: str              # System prompt
    user_input: str
    query_type: str
    decision: str
    output: str


# In[40]:


# Schema for strctured output to use as routing logic
class Route(BaseModel):
	step: Literal["finance", "portfolio ", "market", "goal", "news", "tax"] = Field(
		None, description="The next step in the routing process"
	)

# Agument the LLM with schema for structured output
router = llm.with_structured_output(Route)


# In[41]:


def debug_print(message, level="INFO"):
    """Print debug messages with timestamp and level."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


# In[42]:


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


# In[43]:


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


# In[47]:


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


# In[48]:


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


# In[50]:


#state = router_workflow.invoke({"user_input": "How can I create a budget to save for a down payment on a house?"})
#print(state["output"])


# In[ ]:


#state = router_workflow.invoke({"user_input": "what is goal planning?"})
#print(state["output"])


# In[ ]:


#state = router_workflow.invoke({"input": "what is market analysis?"})
#print(state["output"])


# In[ ]:


#state = router_workflow.invoke({"user_input": "what is the news?"})
#print(state["output"])


# In[ ]:


#state = router_workflow.invoke({"user_input": "Can you provide tax education?"})
#print(state["output"])

