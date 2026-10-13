"""
Agentic RAG coaching graph.

router -> retrieve -> grader -> (YES) -> generate -> END
                                 -> (NO)  -> broaden (re-retrieve, wider query) -> generate -> END

The LLM never invents advice: it only rewrites the retrieved coaching-manual
passage into ONE short, plain-language sentence, and the output is checked by
Python rules (length, banned jargon, one sentence). If the check fails twice,
a hand-written fallback tip is used instead.
"""
import re
from typing import TypedDict

from langgraph.graph import StateGraph, END
from rag_engine import vectorstore, llm, feedback_cache


class AgentState(TypedDict, total=False):
    error_type: str          # e.g. "Arc_Too_Flat" or "Arc_Too_Flat_1" (both accepted)
    query: str               # retrieval query (optional)
    focus: str               # what the tip should tell the player to do (optional)
    retrieved_content: str
    source: str
    is_relevant: str
    feedback: str
    used_fallback: bool


# ---------------------------------------------------------------------------
# Plain-language description of each error (what the LLM is told the problem is)
# ---------------------------------------------------------------------------
KNOWN_ERRORS = ["Aim_Off", "Arc_Too_Flat", "Low_Velocity", "Jerky_Motion"]

PROBLEM_TEXT = {
    "Aim_Off":      "the ball is going to the left or right of the hoop",
    "Arc_Too_Flat": "the ball is thrown too flat and too straight, so it does not drop into the hoop",
    "Low_Velocity": "the throw is too weak and the ball does not reach the hoop",
    "Jerky_Motion": "the throwing arm moves in a jerky, stop-and-start way",
}

DEFAULT_QUERY = {
    "Aim_Off":      "keep elbow tucked under the ball and point fingers straight at the basket",
    "Arc_Too_Flat": "raise the elbow and release high so the ball arcs up into the basket",
    "Low_Velocity": "extend the shooting arm fully and push upward for more power",
    "Jerky_Motion": "smooth continuous fluid shooting motion without pausing, relaxed arm",
}

# Hand-written fallbacks, one per variation index (0, 1, 2). Used only if the LLM fails validation.
FALLBACK_TIPS = {
    "Aim_Off": [
        "Keep your elbow under the ball and point your hand at the hoop.",
        "Line your shoulder up with the hoop before you throw.",
        "Keep your elbow close to your body so the ball goes straight.",
    ],
    "Arc_Too_Flat": [
        "Throw the ball higher and aim above the hoop so it drops in.",
        "Lift your elbow up before you throw so the ball arcs higher.",
        "Finish with your hand pointing up, not forward.",
    ],
    "Low_Velocity": [
        "Push your arm all the way up so the ball gets more power.",
        "Throw a little harder and finish with your arm fully straight.",
        "Use your whole arm, not just your hand, to push the ball.",
    ],
    "Jerky_Motion": [
        "Throw in one smooth motion without stopping halfway.",
        "Relax your shoulders and keep your arm moving evenly.",
        "Move your arm up in one flowing line, then let go.",
    ],
}

# Words a beginner will not understand -> tip is rejected if it contains any of these.
BANNED_WORDS = {
    "flexion", "extension", "kinetic", "trajectory", "vertically", "plane", "abduction",
    "biomechanical", "backspin", "apex", "parabolic", "mechanics", "release point",
    "dominant", "lateral", "momentum", "torso", "elevation", "fluid", "optimal",
}

MIN_WORDS, MAX_WORDS = 6, 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def base_error_of(error_type: str) -> str:
    """'Arc_Too_Flat_1' -> 'Arc_Too_Flat'. (The old split('_')[0] turned this into just 'Arc'.)"""
    for name in KNOWN_ERRORS:
        if error_type.startswith(name):
            return name
    return "Aim_Off"


def variant_index_of(error_type: str) -> int:
    m = re.search(r"_(\d+)$", error_type)
    return int(m.group(1)) if m else 0


def clean_tip(raw: str) -> str:
    """Keep only the first sentence of the first non-empty line, with no quotes/bullets/labels."""
    for line in raw.strip().splitlines():
        line = line.strip().strip('"\'“”*-• ').strip()
        line = re.sub(r"^(tip|coach|waspy|answer)\s*[:\-]\s*", "", line, flags=re.I)
        if line:
            m = re.match(r"(.+?[.!?])(\s|$)", line)
            tip = (m.group(1) if m else line).strip()
            if tip and tip[-1] not in ".!?":
                tip += "."
            return tip
    return ""


def validate_tip(tip: str):
    """Returns (ok, reason)."""
    if not tip:
        return False, "empty"
    words = tip.split()
    if len(words) < MIN_WORDS:
        return False, f"too short ({len(words)} words)"
    if len(words) > MAX_WORDS:
        return False, f"too long ({len(words)} words)"
    low = tip.lower()
    for bad in BANNED_WORDS:
        if bad in low:
            return False, f"jargon: {bad}"
    for bad in ("cannot assist", "i can't", "i cannot", "as an ai", "let's", "here is", "here's"):
        if bad in low:
            return False, f"off-script: {bad}"
    if ":" in tip or "\n" in tip:
        return False, "contains label/newline"
    return True, "ok"


def _source_line(chunk_text: str) -> str:
    first = chunk_text.strip().splitlines()[0] if chunk_text.strip() else ""
    return first if first.startswith("SOURCE:") else ""


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def router_node(state: AgentState):
    base = base_error_of(state["error_type"])
    query = state.get("query") or DEFAULT_QUERY[base]
    return {"query": query}


def retrieve_node(state: AgentState):
    docs = vectorstore.similarity_search(state["query"], k=2)
    if not docs:
        return {"retrieved_content": "", "source": ""}
    content = "\n\n".join(d.page_content for d in docs)
    return {"retrieved_content": content, "source": _source_line(docs[0].page_content)}


def grader_node(state: AgentState):
    context = state["retrieved_content"].strip()
    if not context:
        return {"is_relevant": "NO"}

    base = base_error_of(state["error_type"])
    prompt = f"""You are checking evidence for a basketball coaching system.

PROBLEM: {PROBLEM_TEXT[base]}

COACHING TEXT:
{context}

Does the coaching text give advice that would help fix this problem?
Reply with exactly one word: YES or NO."""
    answer = llm.invoke(prompt).content.strip().upper()
    # startswith: a small model often answers "YES." or "YES, because..." which the old == "YES" rejected.
    return {"is_relevant": "YES" if answer.startswith("YES") else "NO"}


def broaden_node(state: AgentState):
    """Grader said the evidence was off-topic: retry with the generic query for this error."""
    base = base_error_of(state["error_type"])
    docs = vectorstore.similarity_search(DEFAULT_QUERY[base], k=2)
    content = "\n\n".join(d.page_content for d in docs)
    return {"retrieved_content": content,
            "source": _source_line(docs[0].page_content) if docs else ""}


def generate_node(state: AgentState):
    base = base_error_of(state["error_type"])
    idx = variant_index_of(state["error_type"])
    focus = state.get("focus", "")

    prompt = f"""You are Waspy, a friendly basketball coach for complete beginners in a VR game.

The player's problem: {PROBLEM_TEXT[base]}.
{("What the tip should be about: " + focus) if focus else ""}

Coaching manual text (use ONLY ideas from this text):
{state["retrieved_content"]}

Write ONE coaching tip.
- One complete sentence, 8 to 14 words.
- Tell the player ONE thing to do on the next throw, and why it helps.
- Use simple everyday words, as if talking to a child. Do not use technical terms.
- Do not mention legs, knees or feet. Only talk about the arm, hand, elbow, shoulder or the ball.

Examples of the style (different topics):
- Keep your eyes on the hoop so your throw goes where you look.
- Hold your hand up at the end so the ball keeps flying straight.

Write only the tip sentence."""

    tip, used_fallback, reason = "", True, "no attempt"
    for attempt in range(3):
        try:
            candidate = clean_tip(llm.invoke(prompt).content)
        except Exception as e:                      # Ollama down, timeout, etc.
            candidate, reason = "", f"llm error: {e}"
            break
        ok, reason = validate_tip(candidate)
        if ok:
            tip, used_fallback = candidate, False
            break
        print(f"   [retry {attempt + 1}] rejected ({reason}): {candidate!r}")

    if used_fallback:
        pool = FALLBACK_TIPS[base]
        tip = pool[idx % len(pool)]

    flag = "FALLBACK" if used_fallback else "LLM"
    print(f"[{state['error_type']} -> {flag}]: {tip}")
    if state.get("source"):
        print(f"   evidence: {state['source']}")

    feedback_cache[state["error_type"]] = tip
    return {"feedback": tip, "used_fallback": used_fallback}


def route_after_grade(state: AgentState):
    return "generate" if state["is_relevant"] == "YES" else "broaden"


# ---------------------------------------------------------------------------
# GRAPH
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grader", grader_node)
workflow.add_node("broaden", broaden_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("router")
workflow.add_edge("router", "retrieve")
workflow.add_edge("retrieve", "grader")
workflow.add_conditional_edges("grader", route_after_grade, {"generate": "generate", "broaden": "broaden"})
workflow.add_edge("broaden", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()