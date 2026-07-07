"""
LangGraph workflow for the AI Resume Analyzer.

Pipeline:
    parse_resume  ->  analyze_match  ->  score_resume  ->  generate_feedback

Each node calls an LLM (Groq or OpenAI, chosen by the user) and passes
structured results forward through a shared State object.
"""

import json
import re
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# 1. Shared state definition
# ---------------------------------------------------------------------------
class ResumeState(TypedDict, total=False):
    resume_text: str
    job_description: str

    llm: object  # the bound chat model instance (Groq or OpenAI)

    candidate_profile: dict
    skills_match: dict
    match_score: int
    score_breakdown: dict
    strengths: List[str]
    weaknesses: List[str]
    feedback: str
    improved_bullets: List[str]

    error: Optional[str]


# ---------------------------------------------------------------------------
# Helper: robustly pull JSON out of an LLM response
# ---------------------------------------------------------------------------
def _extract_json(raw_text: str) -> dict:
    """LLMs sometimes wrap JSON in markdown fences or add commentary. Clean it."""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    # try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # fallback: grab the largest {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model output:\n{raw_text[:500]}")


def _call_llm(llm, system_prompt: str, user_prompt: str) -> str:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# 2. Node: parse the resume into a structured profile
# ---------------------------------------------------------------------------
def parse_resume(state: ResumeState) -> ResumeState:
    system_prompt = (
        "You are an expert technical recruiter. Extract structured information "
        "from the resume text you are given. Respond with STRICT JSON only, "
        "no markdown, no commentary, matching exactly this schema:\n"
        "{\n"
        '  "name": string,\n'
        '  "email": string,\n'
        '  "years_experience": number,\n'
        '  "skills": [string],\n'
        '  "education": [string],\n'
        '  "work_experience": [string],\n'
        '  "summary": string\n'
        "}"
    )
    user_prompt = f"Resume text:\n\n{state['resume_text']}"

    raw = _call_llm(state["llm"], system_prompt, user_prompt)
    profile = _extract_json(raw)
    return {"candidate_profile": profile}


# ---------------------------------------------------------------------------
# 3. Node: compare resume against the job description (if provided)
# ---------------------------------------------------------------------------
def analyze_match(state: ResumeState) -> ResumeState:
    jd = state.get("job_description", "").strip()
    profile = state["candidate_profile"]

    if not jd:
        # No JD provided -> general skills analysis only
        system_prompt = (
            "You are a career coach. Given a candidate profile JSON, identify "
            "their top skills and any notable gaps for a strong resume in their "
            "field. Respond with STRICT JSON only:\n"
            "{\n"
            '  "matched_skills": [string],\n'
            '  "missing_skills": [string],\n'
            '  "extra_skills": [string]\n'
            "}"
        )
        user_prompt = f"Candidate profile:\n{json.dumps(profile)}"
    else:
        system_prompt = (
            "You are an ATS (Applicant Tracking System) and expert recruiter. "
            "Compare the candidate profile JSON against the job description. "
            "Respond with STRICT JSON only:\n"
            "{\n"
            '  "matched_skills": [string],\n'
            '  "missing_skills": [string],\n'
            '  "extra_skills": [string]\n'
            "}"
        )
        user_prompt = (
            f"Candidate profile:\n{json.dumps(profile)}\n\n"
            f"Job description:\n{jd}"
        )

    raw = _call_llm(state["llm"], system_prompt, user_prompt)
    skills_match = _extract_json(raw)
    return {"skills_match": skills_match}


# ---------------------------------------------------------------------------
# 4. Node: produce a numeric match/quality score with breakdown
# ---------------------------------------------------------------------------
def score_resume(state: ResumeState) -> ResumeState:
    jd = state.get("job_description", "").strip()
    profile = state["candidate_profile"]
    skills_match = state["skills_match"]

    system_prompt = (
        "You are a strict but fair ATS scoring engine. Score the resume from "
        "0-100 based on skills match, experience relevance, clarity, and "
        "keyword optimization. Respond with STRICT JSON only:\n"
        "{\n"
        '  "overall_score": number,\n'
        '  "breakdown": {\n'
        '    "skills_match": number,\n'
        '    "experience_relevance": number,\n'
        '    "clarity_and_formatting": number,\n'
        '    "keyword_optimization": number\n'
        "  },\n"
        '  "strengths": [string],\n'
        '  "weaknesses": [string]\n'
        "}\n"
        "Each breakdown value must be 0-25 and sum to overall_score."
    )
    user_prompt = (
        f"Candidate profile:\n{json.dumps(profile)}\n\n"
        f"Skills match analysis:\n{json.dumps(skills_match)}\n\n"
        f"Job description (may be empty):\n{jd}"
    )

    raw = _call_llm(state["llm"], system_prompt, user_prompt)
    result = _extract_json(raw)

    return {
        "match_score": result.get("overall_score", 0),
        "score_breakdown": result.get("breakdown", {}),
        "strengths": result.get("strengths", []),
        "weaknesses": result.get("weaknesses", []),
    }


# ---------------------------------------------------------------------------
# 5. Node: generate human-readable feedback + improved bullet points
# ---------------------------------------------------------------------------
def generate_feedback(state: ResumeState) -> ResumeState:
    system_prompt = (
        "You are a supportive but honest professional resume coach. Given the "
        "candidate's profile, weaknesses, and score breakdown, write:\n"
        "1. A short (4-6 sentence) actionable feedback paragraph in plain text.\n"
        "2. 3-5 rewritten/improved resume bullet points that fix the weaknesses.\n\n"
        "Respond with STRICT JSON only:\n"
        "{\n"
        '  "feedback": string,\n'
        '  "improved_bullets": [string]\n'
        "}"
    )
    user_prompt = (
        f"Candidate profile:\n{json.dumps(state['candidate_profile'])}\n\n"
        f"Weaknesses:\n{json.dumps(state.get('weaknesses', []))}\n\n"
        f"Score breakdown:\n{json.dumps(state.get('score_breakdown', {}))}"
    )

    raw = _call_llm(state["llm"], system_prompt, user_prompt)
    result = _extract_json(raw)

    return {
        "feedback": result.get("feedback", ""),
        "improved_bullets": result.get("improved_bullets", []),
    }


# ---------------------------------------------------------------------------
# 6. Build and compile the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ResumeState)

    graph.add_node("parse_resume", parse_resume)
    graph.add_node("analyze_match", analyze_match)
    graph.add_node("score_resume", score_resume)
    graph.add_node("generate_feedback", generate_feedback)

    graph.add_edge(START, "parse_resume")
    graph.add_edge("parse_resume", "analyze_match")
    graph.add_edge("analyze_match", "score_resume")
    graph.add_edge("score_resume", "generate_feedback")
    graph.add_edge("generate_feedback", END)

    return graph.compile()


def run_resume_analysis(llm, resume_text: str, job_description: str = "") -> dict:
    """Convenience wrapper: build the graph, run it, return the final state."""
    app = build_graph()
    initial_state: ResumeState = {
        "llm": llm,
        "resume_text": resume_text,
        "job_description": job_description,
    }
    final_state = app.invoke(initial_state)
    return final_state
