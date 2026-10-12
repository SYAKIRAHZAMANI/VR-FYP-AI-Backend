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
    # Retrieve several passages instead of relying on only one.
    docs = retriever.invoke(state["error_type"])

    if not docs:
        return {"retrieved_content": ""}

    content = "\n\n".join(
        doc.page_content for doc in docs[:3]
    )

    return {"retrieved_content": content}


def grader_node(state: AgentState):
    context = state["retrieved_content"].strip()

    if not context:
        return {"is_relevant": "NO"}

    prompt = f"""
You are checking evidence for a basketball coaching system.

DETECTED ERROR:
{state["error_type"]}

RETRIEVED TEXT FROM THE FIBA MANUAL:
{context}

Does the retrieved text contain useful advice directly
related to correcting the detected error?

Answer YES only if the text provides relevant evidence.
Otherwise answer NO.

Return exactly one word: YES or NO.
"""

    response = llm.invoke(prompt)
    answer = response.content.strip().upper()

    # Accept only an exact YES response.
    if answer == "YES":
        return {"is_relevant": "YES"}

    return {"is_relevant": "NO"}

# --- NODE: Generate ---
def generate_node(state: AgentState):
    if state["is_relevant"] == "SKIP":
        return {}

    if "YES" in state["is_relevant"]:
        context = state["retrieved_content"]
    else:
        broad_docs = vectorstore.similarity_search(state["error_type"], k=2)
        context = "\n".join(d.page_content for d in broad_docs)

    base_error = state['error_type'].split('_')[0].strip()

    # A blunt, direct prompt for a small model
    prompt = f"""Task: Write a 5-word basketball coaching command to fix a {base_error}.
Context: {context}
Rules: Exactly 5 words. No intro. No explanation. Just the command."""

    try:
        response = llm.invoke(prompt)
        raw = response.content.replace('"', '').strip()
        
        # If the LLM starts talking like a toddler or trips a safety filter, trigger fallback
        if "cannot assist" in raw.lower() or "let's learn" in raw.lower() or "first" in raw.lower():
            raise Exception("LLM went off the rails")

        # --- PYTHON FORCE-TRUNCATION SAFETY NET ---
        # Even if the AI writes an essay, Python physically lops it off at 8 words!
        words = raw.split()
        if len(words) > 8:
            clean_feedback = " ".join(words[:8])
        else:
            clean_feedback = raw
            
    except Exception:
        # Bulletproof athletic fallbacks
        fallbacks = {
            "Aim_Off": "Point your hand at the hoop.",
            "Arc_Too_Flat": "Release the ball much higher.",
            "Low_Velocity": "Fully extend your arm to throw.",
            "Jerky_Motion": "Keep your arm motion smooth."
        }
        clean_feedback = fallbacks.get(base_error, "Keep your arm smooth.")

    print(f"\n[{base_error} -> FORCED PUNCHY TIP]: {clean_feedback}\n")

    feedback_cache[state["error_type"]] = clean_feedback
    return {"feedback": clean_feedback}
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