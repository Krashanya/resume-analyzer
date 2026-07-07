# 📄 AI Resume Analyzer

An AI-powered resume analyzer built with **LangGraph**, **LangChain**, **Groq**, and **OpenAI**, deployed with **Streamlit**.

Upload a resume (PDF/TXT), optionally paste a job description, and get:
- A structured candidate profile (skills, experience, education)
- A skills-match analysis (matched / missing / extra skills)
- An ATS-style score (0–100) with a breakdown
- Strengths & weaknesses
- Actionable coaching feedback
- Rewritten, improved resume bullet points
- A downloadable JSON report

---

## 🗂️ Project Structure

```
resume_analyzer/
├── app.py                  # Streamlit UI (entry point)
├── graph.py                # LangGraph pipeline (the "AI brain")
├── utils.py                # PDF/text extraction helpers
├── requirements.txt        # Python dependencies
├── .env.example            # Template for local API keys
├── .gitignore
├── .streamlit/
│   └── config.toml         # Theme settings
└── README.md
```

### How the pipeline works (`graph.py`)

```
START → parse_resume → analyze_match → score_resume → generate_feedback → END
```

1. **parse_resume** – LLM extracts structured JSON (name, skills, experience, education, summary) from raw resume text.
2. **analyze_match** – Compares the parsed profile against the job description (or does a general skills review if no JD is given).
3. **score_resume** – Produces a 0–100 score with a 4-part breakdown (skills match, experience relevance, clarity, keyword optimization).
4. **generate_feedback** – Writes a short coaching paragraph and rewrites 3–5 bullet points to fix weaknesses.

Each node is a Python function that receives the shared `ResumeState`, calls the LLM, and returns updates — this is the core LangGraph pattern.

---

## 🧰 Step 1: Prerequisites

- Python 3.10+ installed
- A free **Groq** API key: https://console.groq.com/keys
- (Optional) An **OpenAI** API key: https://platform.openai.com/api-keys
- A GitHub account (for deployment)

---

## 💻 Step 2: Run Locally

1. **Unzip the project** and open a terminal in the `resume_analyzer` folder.

2. **Create a virtual environment:**
   ```bash
   python -m venv venv

   # Activate it:
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API keys.** Rename `.env.example` to `.env` and fill it in:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
   ```
   (You only need to fill in the provider you plan to use — the app also lets you paste a key directly into the sidebar at runtime, so the `.env` file is optional.)

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

6. Your browser should open automatically at `http://localhost:8501`. If not, open that URL manually.

7. In the sidebar: choose **Groq** or **OpenAI**, confirm/enter your API key, pick a model, then upload a resume and click **Analyze Resume**.

---

## ☁️ Step 3: Deploy to Streamlit Community Cloud (Free)

1. **Push the project to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AI Resume Analyzer"
   git branch -M main
   git remote add origin https://github.com/<your-username>/resume-analyzer.git
   git push -u origin main
   ```
   > `.env` is already excluded via `.gitignore` — never commit real API keys.

2. Go to **https://share.streamlit.io** and sign in with GitHub.

3. Click **"New app"**, then select:
   - **Repository:** `<your-username>/resume-analyzer`
   - **Branch:** `main`
   - **Main file path:** `app.py`

4. Click **"Advanced settings"** → **Secrets**, and add your keys in TOML format:
   ```toml
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
   OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxx"
   ```
   This is the cloud equivalent of your local `.env` file — the app already checks `st.secrets` first, so it'll pick these up automatically and pre-fill the sidebar.

5. Click **"Deploy"**. Streamlit will install `requirements.txt` and launch your app. First build takes 1–3 minutes.

6. You'll get a public URL like:
   ```
   https://resume-analyzer-<random>.streamlit.app
   ```
   Share it with anyone!

---

## 🔁 Step 4: Updating the Deployed App

Any time you push new commits to the `main` branch on GitHub, Streamlit Cloud automatically redeploys:
```bash
git add .
git commit -m "Update analysis logic"
git push
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: langgraph` | Run `pip install -r requirements.txt` again inside your activated venv |
| Blank/empty resume text extracted | Your PDF may be a scanned image — try a text-based PDF or a `.txt` file |
| `AuthenticationError` from Groq/OpenAI | Double-check the API key is correct and has no extra spaces |
| JSON parsing errors in the app | Usually a transient LLM formatting hiccup — click "Analyze Resume" again; lowering `temperature` in `graph.py` also helps |
| App works locally but not on Streamlit Cloud | Make sure secrets are added under **Advanced settings → Secrets**, not just your local `.env` |

---

## 🚀 Ideas to Extend This Project

- Add support for `.docx` resumes (using `python-docx`)
- Add a "compare multiple resumes against one JD" batch mode
- Cache LLM responses with `st.cache_data` to reduce API costs
- Add authentication (Streamlit supports simple auth via `st.secrets` + a login gate)
- Store analysis history in a database (e.g., SQLite or Supabase)
- Add a LangGraph conditional edge that re-runs feedback generation if the score is very low

---

## 📜 License

Free to use and modify for personal or commercial projects.
