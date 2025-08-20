from typing import Dict, List, TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    query: str
    agent_name: str
    decision: str
    rag_context: List[str]
    response: Dict[str, any]
    messages: Annotated[List[BaseMessage], add_messages]
