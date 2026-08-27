from langgraph.graph import StateGraph, END
from typing import TypedDict
from rag_engine import retriever, vectorstore, llm, feedback_cache


class AgentState(TypedDict):
    error_type: str
    retrieved_content: str
    is_relevant: str
    feedback: str


# --- NODE: Router (entry point) ---
# Currently a pass-through - kept as its own node (rather than folded into
# retrieve) so future routing logic (e.g. skip RAG entirely for very minor
# errors) has a clear place to live without restructuring the graph.
def router_node(state: AgentState):
    return {}


# --- NODE: Retrieve ---
def retrieve_node(state: AgentState):
    # Check cache first - skip retrieval + LLM calls entirely if we've
    # already produced feedback for this exact error type this session.
    if state["error_type"] in feedback_cache:
        print(f"[Cache hit] Returning cached feedback for '{state['error_type']}'")
        return {"feedback": feedback_cache[state["error_type"]], "retrieved_content": "CACHED"}

    docs = retriever.invoke(state["error_type"])
    content = docs[0].page_content if docs else ""
    return {"retrieved_content": content}


# --- NODE: CRAG Grader ---
def grader_node(state: AgentState):
    # If retrieve_node already served a cached answer, skip grading.
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
        # Already have feedback from cache - nothing to generate.
        return {}

    if "YES" in state["is_relevant"]:
        context = state["retrieved_content"]
    else:
        # CRAG fallback: initial retrieval graded as not relevant enough,
        # so broaden the search instead of trusting a single weak match.
        print("Retrieval grade low, attempting broader search...")
        broad_docs = vectorstore.similarity_search(state["error_type"], k=3)
        context = "\n".join(d.page_content for d in broad_docs)

    prompt = f"Using {context}, give a 10-word coaching tip for {state['error_type']}."
    feedback = llm.invoke(prompt).content.strip()

    feedback_cache[state["error_type"]] = feedback
    return {"feedback": feedback}


# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grader", grader_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("router")
workflow.add_edge("router", "retrieve")   # <-- this edge was missing before
workflow.add_edge("retrieve", "grader")
workflow.add_edge("grader", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


if __name__ == "__main__":
    print("--- TESTING LANGGRAPH PIPELINE ---")

    print("\n--- Test 1: Known Error ---")
    result1 = app.invoke({"error_type": "Early_Release"})
    print(f"Feedback: {result1['feedback']}")

    print("\n--- Test 2: Same Error Again (should hit cache) ---")
    result2 = app.invoke({"error_type": "Early_Release"})
    print(f"Feedback: {result2['feedback']}")

    print("\n--- Test 3: Unknown Error (forces broad-search fallback) ---")
    result3 = app.invoke({"error_type": "ERROR: Alien_Abduction"})
    print(f"Feedback: {result3['feedback']}")