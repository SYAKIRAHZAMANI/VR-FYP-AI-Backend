from langgraph.graph import StateGraph, END
from typing import TypedDict

# This "State" keeps track of the data moving through our agent
class AgentState(TypedDict):
    error_type: str
    retrieved_content: str
    feedback: str
    confidence: float

# The "Intelligent Router"
def query_router(state: AgentState):
    # Logic: If the error is critical, use RAG; if minor, use simple logic
    if "Release" in state['error_type']:
        return "RAG_FLOW"
    return "SIMPLE_LOGIC"

# The "RAG Retriever" (Your existing logic)
def retrieve_expert_advice(state: AgentState):
    # Use your chroma retriever here
    docs = retriever.invoke(state['error_type'])
    return {"retrieved_content": docs[0].page_content}

# The "Generator"
def generate_feedback(state: AgentState):
    prompt = f"Coaching advice for {state['error_type']} based on {state['retrieved_content']}"
    # call your LLM here...
    return {"feedback": "Actionable feedback string"}

# Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("router", query_router)
workflow.add_node("rag", retrieve_expert_advice)
workflow.add_node("generator", generate_feedback)

workflow.set_entry_point("router")
workflow.add_edge("rag", "generator")
workflow.add_edge("generator", END)

# Compile
app = workflow.compile()