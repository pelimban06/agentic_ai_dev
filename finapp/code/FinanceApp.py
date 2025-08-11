#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install langchain langchain-openai')
get_ipython().system('pip install langgraph')


# In[7]:


from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


# In[8]:


import os
import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# In[9]:


from langchain_core.prompts import PromptTemplate


# In[12]:


from langchain_openai import ChatOpenAI
# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model_name="gpt-4o",  # Specify the model (e.g., gpt-4, gpt-3.5-turbo)
    temperature=0.7,      # Controls randomness (0 to 1)
    max_tokens=512        # Maximum tokens in the response
)


# In[15]:


# State
class State(TypedDict):
	input: str
	decision: str
	output: str


# In[76]:


from typing_extensions import Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


# Schema for strctured output to use as routing logic
class Route(BaseModel):
	step: Literal["finance", "portfolio ", "market", "goal", "news", "tax"] = Field(
		None, description="The next step in the routing prcess"
	)

# Agument the LLM with schema for structured output

router = llm.with_structured_output(Route)


# In[77]:


from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages import SystemMessage, HumanMessage
# Nodes
def call_finance_qa_agent(state: State):
	""" Finance Q&A Agent """

	print(" Finance Q&A Agent: Handles general financial education queries ")
	result = llm.invoke(state["input"])
	return {"output": result.content}

def call_portfolio_analysis_agent(state: State):
	""" Portfolio Analysis Agent """

	print(" Reviews and analyzes user portfolios ")
	result = llm.invoke(state["input"])
	return {"output": result.content}

def call_market_analysis_agent(state: State):
	""" Market Analysis Agent"""

	print(" Provides real-time market insights ")
	result = llm.invoke(state["input"])
	return {"output": result.content}

def call_goal_planning_agent(state: State):
	""" Goal Planning Agent"""

	print(" Assists with financial goal setting and planning ")
	result = llm.invoke(state["input"])
	return {"output": result.content}

def call_news_synthesizer_agent(state: State):
	""" News Synthesizer Agent"""

	print(" Summarizes and contextualizes financial news ")
	result = llm.invoke(state["input"])
	return {"output": result.content}

def call_tax_education_agent(state: State):
	""" Tax Education Agent"""

	print(" Explains tax concepts and account types ")
	result = llm.invoke(state["input"])
	return {"output": result.content}


def finapp_router(state: State):
  """Route the input to the appropriate node """

  # Run the agumented LLM with strctured output to serve as routing logic
  decision = router.invoke(
		[
			SystemMessage(
				content="Route the input to finance, portfolio, market, goal, news or tax based on user request"
			),
			HumanMessage(content=state["input"]),

		]
	)
  print(f"decision: {decision.step}")  
  return {"decision": decision.step}


# Conditional edge function to route the appropriate node
def route_decision(state: State):
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


# In[78]:


# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("call_finance_qa_agent", call_finance_qa_agent)
router_builder.add_node("call_portfolio_analysis_agent", call_portfolio_analysis_agent)
router_builder.add_node("call_market_analysis_agent", call_market_analysis_agent)
router_builder.add_node("call_goal_planning_agent", call_goal_planning_agent)
router_builder.add_node("call_news_synthesizer_agent", call_news_synthesizer_agent)
router_builder.add_node("call_tax_education_agent", call_tax_education_agent)

router_builder.add_node("finapp_router", finapp_router)

# Add edges to connect nodes
router_builder.add_edge(START, "finapp_router")
router_builder.add_conditional_edges(
    "finapp_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "call_finance_qa_agent": "call_finance_qa_agent",
        "call_portfolio_analysis_agent": "call_portfolio_analysis_agent",
        "call_market_analysis_agent": "call_market_analysis_agent",
        "call_goal_planning_agent": "call_goal_planning_agent",
        "call_news_synthesizer_agent": "call_news_synthesizer_agent",
        "call_tax_education_agent": "call_tax_education_agent",
    },
)

router_builder.add_edge("call_finance_qa_agent", END)
router_builder.add_edge("call_portfolio_analysis_agent", END)
router_builder.add_edge("call_market_analysis_agent", END)
router_builder.add_edge("call_goal_planning_agent", END)
router_builder.add_edge("call_news_synthesizer_agent", END)
router_builder.add_edge("call_tax_education_agent", END)

# compile workflow
router_workflow = router_builder.compile()

# Show the workflow
display(Image(router_workflow.get_graph().draw_mermaid_png()))


# In[79]:


state = router_workflow.invoke({"input": "what is portfolio?"})
print(state["output"])


# In[80]:


state = router_workflow.invoke({"input": "what is finance?"})
print(state["output"])


# In[81]:


state = router_workflow.invoke({"input": "what is goal planning?"})
print(state["output"])


# In[82]:


state = router_workflow.invoke({"input": "what is market analysis?"})
print(state["output"])


# In[83]:


state = router_workflow.invoke({"input": "what is the news?"})
print(state["output"])


# In[84]:


state = router_workflow.invoke({"input": "Can you provide tax education?"})
print(state["output"])


# In[ ]:




