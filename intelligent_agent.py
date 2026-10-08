from langgraph.graph import StateGraph, END
from typing import TypedDict
from rag_engine import retriever, vectorstore, llm, feedback_cache

class AgentState(TypedDict):
    error_type: str
    retrieved_content: str
    is_relevant: str
    feedback: str

# --- NODE: Router (entry point) ---
def router_node(state: AgentState):
    return {}

# --- NODE: Retrieve ---
def retrieve_node(state: AgentState):
    if state["error_type"] in feedback_cache:
        return {"feedback": feedback_cache[state["error_type"]], "retrieved_content": "CACHED"}

    docs = retriever.invoke(state["error_type"])
    content = docs[0].page_content if docs else ""
    return {"retrieved_content": content}

# --- NODE: CRAG Grader ---
def grader_node(state: AgentState):
    if state["retrieved_content"] == "CACHED":
        return {"is_relevant": "SKIP"}

    prompt = (
        f"You are a coach's assistant. Is the retrieved context relevant "
        f"to fixing this error?\nError: {state['error_type']}\n"
        f"Context: {state['retrieved_content']}\nAnswer ONLY 'YES' or 'NO'."
    )
    response = llm.invoke(prompt)
    return {"is_relevant": response.content.strip().upper()}

# --- NODE: Generate ---
def generate_node(state: AgentState):
    if state["is_relevant"] == "SKIP":
        return {}

    if "YES" in state["is_relevant"]:
        context = state["retrieved_content"]
    else:
        print("Retrieval grade low, attempting broader search...")
        broad_docs = vectorstore.similarity_search(state["error_type"], k=3)
        context = "\n".join(d.page_content for d in broad_docs)

    # THE HYPE COACH PROMPT GOES HERE!
    prompt = f"""You are a supportive, professional basketball shooting coach.
Read this official coaching manual: {context}

The player just made this error: {state['error_type']}. 
Based ONLY on the manual, give the player ONE short, helpful coaching cue. 
RULES:
1. Maximum 8 words!
2. Do not yell or use all-caps. Be encouraging but calm.
3. Translate the biomechanics into a simple, practical tip.
4. Do not include the error name in your response."""

    feedback = llm.invoke(prompt).content.strip()
    
    # Print it out to the console so you can see what it generated!
    print(f"\n[AI GENERATED TIP]: {feedback}\n")

    feedback_cache[state["error_type"]] = feedback
    return {"feedback": feedback}

# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grader", grader_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("router")
workflow.add_edge("router", "retrieve")   
workflow.add_edge("retrieve", "grader")
workflow.add_edge("grader", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()