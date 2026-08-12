from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage, ToolMessage

"""
ToolRuntime (from langchain.tools import ToolRuntime) gets auto-injected into any tool function when a parameter is named runtime and type-hinted as ToolRuntime — no Annotated wrapper needed. It exposes:

	• state — the current graph state (e.g. runtime.state["messages"])
	• context — static run-time context you passed to agent.invoke(..., context=...), typed by your context_schema (e.g. user ID, DB connections, config)
	• config — the RunnableConfig for the current execution
	• store — a BaseStore instance for long-term memory (runtime.store.get(...), .put(...))
	• stream_writer — writer for pushing custom updates on the "custom" stream mode (progress, partial output, etc.)
	• tool_call_id — the ID of the current tool call
	• tools — the list of all BaseTool instances available in the agent
	• execution_info — thread ID, run ID, attempt number for this execution (requires deepagents>=0.5.0 or langgraph>=1.1.5)
	• server_info — server-specific metadata when running on LangGraph Server (assistant ID, graph ID, authenticated user); None in local dev

"""

