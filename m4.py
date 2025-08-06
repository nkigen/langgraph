import os
from typing import Annotated
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool


from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt
import json

from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tool=TavilySearch(max_results=2)
tools=[tool, human_assistance]


llm = init_chat_model("openai:gpt-4-turbo")

llm_with_tools=llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    
    assert(len(message.tool_calls)) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)

# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)

# graph = graph_builder.compile()



tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"

config = {"configurable": {"thread_id":"1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config=config,
    stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()

human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})
events = graph.stream(human_command, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
