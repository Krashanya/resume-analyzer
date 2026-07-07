"""
AI Resume Analyzer
-------------------
Streamlit front-end. Lets the user upload a resume, optionally paste a job
description, and runs a LangGraph pipeline that parses, scores, and gives
feedback on the resume. Provider/model/API key are configured by the
developer only (see "Backend configuration" below) - end users never see
or enter any key.
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv

from utils import extract_text_from_upload, truncate_text
from graph import run_resume_analysis

load_dotenv()  # loads .env for local development

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Backend configuration (NOT shown to end users)
# ---------------------------------------------------------------------------
# The provider, model, and API key are all configured by the developer here
# and pulled from Streamlit secrets / a local .env file. End users never see
# or enter any key - they just use the app.
#
# To change provider/model, edit the two lines below.
PROVIDER = "Groq"                          # "Groq" or "OpenAI"
MODEL_NAME = "llama-3.3-70b-versatile"     # e.g. "gpt-4o-mini" if using OpenAI


def get_secret(env_var: str) -> str:
    """Preference order: Streamlit secrets (cloud) -> local .env file."""
    try:
        return st.secrets[env_var]
    except Exception:
        return os.getenv(env_var, "")


provider = PROVIDER
model_name = MODEL_NAME
api_key = get_secret("GROQ_API_KEY" if provider == "Groq" else "OPENAI_API_KEY")

st.sidebar.title("📄 About")
st.sidebar.write(
    "This tool analyzes your resume using AI and gives you an ATS-style score, "
    "skill-gap analysis, and improved bullet points."
)
st.sidebar.caption("🔒 No API key or personal data is stored.")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("📄 AI Resume Analyzer")
st.write(
    "Upload your resume and (optionally) paste a job description. "
    "Get an ATS-style score, skill-gap analysis, and rewritten bullet points — "
    "powered by LangGraph + LangChain."
)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF or TXT)", type=["pdf", "txt"]
    )

with col2:
    job_description = st.text_area(
        "Paste the job description (optional)",
        height=200,
        placeholder="Paste the job posting here for a tailored match score...",
    )

analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Build the LLM instance for the chosen provider
# ---------------------------------------------------------------------------
def get_llm(provider: str, api_key: str, model_name: str):
    if provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=api_key, model=model_name, temperature=0.3)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=0.3)


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not api_key:
        st.error(
            f"⚠️ No {provider} API key is configured on the server. "
            "If you're the developer: add it to `.env` (local) or "
            "Streamlit Cloud → Settings → Secrets (deployed)."
        )
        st.stop()

    if not uploaded_file:
        st.error("Please upload a resume file first.")
        st.stop()

    with st.spinner("Extracting resume text..."):
        try:
            resume_text = extract_text_from_upload(uploaded_file)
            resume_text = truncate_text(resume_text)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

    if not resume_text.strip():
        st.error("Couldn't extract any text from that file. Try a different resume file.")
        st.stop()

    try:
        llm = get_llm(provider, api_key, model_name)
    except Exception as e:
        st.error(f"Failed to initialize {provider} model: {e}")
        st.stop()

    with st.spinner("Running LangGraph analysis pipeline (parsing → matching → scoring → feedback)..."):
        try:
            result = run_resume_analysis(llm, resume_text, job_description)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    st.success("Analysis complete!")

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------
    profile = result.get("candidate_profile", {})
    skills_match = result.get("skills_match", {})
    score = result.get("match_score", 0)
    breakdown = result.get("score_breakdown", {})
    strengths = result.get("strengths", [])
    weaknesses = result.get("weaknesses", [])
    feedback = result.get("feedback", "")
    improved_bullets = result.get("improved_bullets", [])

    st.markdown("## 📊 Overall Score")
    score_col, breakdown_col = st.columns([1, 2])
    with score_col:
        st.metric("Resume Score", f"{score}/100")
        st.progress(min(max(int(score), 0), 100) / 100)
    with breakdown_col:
        if breakdown:
            for k, v in breakdown.items():
                label = k.replace("_", " ").title()
                st.write(f"**{label}:** {v}/25")
                st.progress(min(max(int(v), 0), 25) / 25)

    st.markdown("---")
    st.markdown("## 👤 Candidate Profile")
    profile_col1, profile_col2 = st.columns(2)
    with profile_col1:
        st.write(f"**Name:** {profile.get('name', 'N/A')}")
        st.write(f"**Email:** {profile.get('email', 'N/A')}")
        st.write(f"**Years of Experience:** {profile.get('years_experience', 'N/A')}")
    with profile_col2:
        st.write(f"**Skills:** {', '.join(profile.get('skills', [])) or 'N/A'}")
    st.write(f"**Summary:** {profile.get('summary', 'N/A')}")

    st.markdown("---")
    st.markdown("## 🎯 Skills Match")
    skill_col1, skill_col2, skill_col3 = st.columns(3)
    with skill_col1:
        st.success("✅ Matched Skills")
        for s in skills_match.get("matched_skills", []):
            st.write(f"- {s}")
    with skill_col2:
        st.warning("⚠️ Missing Skills")
        for s in skills_match.get("missing_skills", []):
            st.write(f"- {s}")
    with skill_col3:
        st.info("➕ Extra Skills")
        for s in skills_match.get("extra_skills", []):
            st.write(f"- {s}")

    st.markdown("---")
    strengths_col, weaknesses_col = st.columns(2)
    with strengths_col:
        st.markdown("### 💪 Strengths")
        for s in strengths:
            st.write(f"- {s}")
    with weaknesses_col:
        st.markdown("### 🔧 Areas to Improve")
        for w in weaknesses:
            st.write(f"- {w}")

    st.markdown("---")
    st.markdown("## 📝 Coach Feedback")
    st.write(feedback)

    st.markdown("## ✍️ Improved Bullet Points")
    for b in improved_bullets:
        st.write(f"- {b}")

    st.markdown("---")
    report = {
        "candidate_profile": profile,
        "skills_match": skills_match,
        "match_score": score,
        "score_breakdown": breakdown,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "feedback": feedback,
        "improved_bullets": improved_bullets,
    }
    st.download_button(
        "⬇️ Download Full Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="resume_analysis_report.json",
        mime="application/json",
    )