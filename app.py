import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai

# Try to import Groq - it's optional, will fall back to Gemini if not available
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH = Path("students_clustered.csv")  # clustered output from notebook
MODEL_PATH = Path("student_model.pkl")      # saved scaler + kmeans + mapping

FEATURE_COLUMNS = [
    "attendance",
    "math_marks",
    "programming_marks",
    "communication_marks",
    "assignment_score",
]

ORDERED_LABELS = ["At Risk", "Average", "Top Performer"]


# ── Data & Model Loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_students() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Could not find data file: {DATA_PATH.resolve()}")
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    # Backwards compatibility: if performance_category missing, fall back gracefully.
    if "performance_category" not in df.columns:
        st.warning(
            "Column 'performance_category' not found. "
            "Please re-run the notebook to generate 'students_clustered.csv'."
        )
    return df


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.exists():
        st.warning(
            f"Model file not found at {MODEL_PATH.resolve()}. "
            "New student prediction will be disabled."
        )
        return None

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    required_keys = {"scaler", "kmeans", "cluster_label_map"}
    if not required_keys.issubset(bundle.keys()):
        st.warning(
            "Model file is missing required keys. "
            "Expected: 'scaler', 'kmeans', 'cluster_label_map'."
        )
        return None

    return bundle


def predict_student_with_model(
    attendance: float,
    math_marks: float,
    programming_marks: float,
    communication_marks: float,
    assignment_score: float,
    bundle,
) -> dict:
    """Mirror the notebook's predict_student helper for live inference."""
    sc = bundle["scaler"]
    km = bundle["kmeans"]
    cmap = bundle["cluster_label_map"]

    features = np.array(
        [[attendance, math_marks, programming_marks, communication_marks, assignment_score]]
    )
    scaled = sc.transform(features)
    cluster = int(km.predict(scaled)[0])
    category = cmap[cluster]

    rules = [
        (programming_marks > 80, "Software Engineer / AI Engineer"),
        (math_marks > 80, "Data Scientist / Finance Analyst"),
        (communication_marks > 80, "Marketing / Business"),
    ]
    careers = ", ".join(label for cond, label in rules if cond) or "General Skill Development"

    return {
        "performance_category": category,
        "career_recommendation": careers,
    }


def _fallback_career_roadmap(student_row: pd.Series, category: str, careers: str) -> list[str]:
    """Rule-based fallback when API unavailable - now more specific."""
    prog = student_row.get('programming_marks', 0)
    math = student_row.get('math_marks', 0)
    comm = student_row.get('communication_marks', 0)
    attendance = student_row.get('attendance', 0)
    
    # Determine primary strength
    scores = {'Programming': prog, 'Math': math, 'Communication': comm}
    primary = max(scores, key=scores.get)
    
    paths = []
    
    # Path 1: Based on primary strength
    if primary == "Programming" and prog >= 60:
        paths.append(
            f"**Software Development Path**: Your coding skills ({prog}/100) are your leverage. "
            f"Complete a Full Stack course (Coursera/Scaler), build 2-3 portfolio projects on GitHub. "
            f"Target service companies (TCS, Infosys, Wipro) for 4-6 LPA starting roles."
        )
    elif primary == "Math" and math >= 60:
        paths.append(
            f"**Data/Analytics Track**: Your analytical thinking ({math}/100) suits data roles. "
            f"Complete Google Data Analytics cert, master Python + SQL basics. "
            f"Target Business Analyst positions at consultancies or startups."
        )
    else:
        paths.append(
            f"**Communication-Led Roles**: Your soft skills ({comm}/100) are valuable. "
            f"Learn basic SQL + Excel, pursue Product Management fundamentals. "
            f"Target non-technical roles at tech companies (customer success, operations)."
        )
    
    # Path 2: Skill development
    weak_areas = []
    if prog < 60:
        weak_areas.append("programming")
    if math < 60:
        weak_areas.append("math")
    if attendance < 75:
        weak_areas.append("attendance")
    
    weak_text = " and ".join(weak_areas) if weak_areas else "all technical skills"
    paths.append(
        f"**Skill Development Focus**: Strengthen {weak_text} through structured learning. "
        f"Use NPTEL, Coursera, or YouTube. Join coding clubs, attend workshops. "
        f"Complete 1-2 industry certifications within 6 months."
    )
    
    # Path 3: Safe backup strategy
    paths.append(
        f"**Mass Recruiter Strategy**: Prepare for volume hiring companies. "
        f"Focus on aptitude tests + basic DSA (Easy level). Attend all campus placement drives. "
        f"Target TCS Digital, Accenture, Cognizant GenC programs for 3.5-5 LPA packages."
    )
    
    return paths


def _generate_with_groq(prompt: str, api_key: str) -> list[str]:
    """Generate roadmap using Groq API (LLaMA models)."""
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Best reasoning model
        messages=[
            {
                "role": "system",
                "content": "You are an expert career counselor for Indian CS/AIML students. Provide concise, actionable, dashboard-ready guidance."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=800,
    )
    
    text = response.choices[0].message.content
    return _parse_ai_response(text)


def _generate_with_gemini(prompt: str, api_key: str) -> list[str]:
    """Generate roadmap using Gemini API."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content(prompt)
    text = response.text or ""
    
    return _parse_ai_response(text)


def _parse_ai_response(text: str) -> list[str]:
    """Parse AI response into exactly 3 roadmap paths."""
    paths = []
    current_path = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Detect path start (bold, numbered, or "Path X")
        is_path_start = (
            line.startswith('**') or 
            line.startswith('Path') or
            (len(paths) < 3 and line.startswith(('1.', '2.', '3.')))
        )
        
        if is_path_start:
            if current_path:
                paths.append(' '.join(current_path))
            current_path = [line]
        else:
            current_path.append(line)
    
    if current_path:
        paths.append(' '.join(current_path))
    
    return paths


def generate_career_roadmap(student_row: pd.Series, category: str, careers: str) -> list[str]:
    """
    Generate 3 personalized career roadmap paths using AI (Groq or Gemini).
    
    Priority: Groq (hackathon stack) > Gemini > Rule-based fallback
    Optimized for dashboard display with concise, actionable output.
    """
    
    # Check which API is available (Groq preferred for hackathon)
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    # If no API keys, use rule-based fallback
    if not groq_key and not gemini_key:
        return _fallback_career_roadmap(student_row, category, careers)
    
    # If Groq SDK not installed but key exists, warn and fall back
    if groq_key and not GROQ_AVAILABLE:
        st.info("💡 Tip: Install Groq for better AI responses: `pip install groq`")
        if not gemini_key:
            return _fallback_career_roadmap(student_row, category, careers)
    
    try:
        # ── ANALYZE STUDENT PROFILE ──
        attendance = student_row.get('attendance', 0)
        math = student_row.get('math_marks', 0)
        prog = student_row.get('programming_marks', 0)
        comm = student_row.get('communication_marks', 0)
        assign = student_row.get('assignment_score', 0)
        interest = student_row.get('career_interest', 'not specified')
        name = student_row.get('name', 'Student')
        
        # Build detailed strength/weakness analysis
        strengths = []
        weaknesses = []
        
        if prog >= 80:
            strengths.append("strong coding skills")
        elif prog < 60:
            weaknesses.append("programming needs work")
        else:
            strengths.append("basic programming foundation")
            
        if math >= 80:
            strengths.append("excellent analytical thinking")
        elif math < 60:
            weaknesses.append("math foundation needs strengthening")
        else:
            strengths.append("decent problem-solving ability")
            
        if comm >= 80:
            strengths.append("outstanding communication")
        elif comm < 60:
            weaknesses.append("soft skills need development")
        else:
            strengths.append("good presentation skills")
            
        if attendance >= 85:
            strengths.append("highly disciplined")
        elif attendance < 75:
            weaknesses.append("consistency issues")
        else:
            strengths.append("regular attendance")
            
        if assign >= 80:
            strengths.append("strong project execution")
        elif assign < 60:
            weaknesses.append("project delivery needs focus")
        else:
            strengths.append("completes assignments")
        
        strength_summary = " • ".join(strengths) if strengths else "building foundational skills"
        weakness_summary = " • ".join(weaknesses) if weaknesses else "no critical gaps"

        # ── BUILD DASHBOARD-OPTIMIZED PROMPT ──
        prompt = f"""You are an AI career counselor for Indian CS/AIML students. Generate a personalized career roadmap for this student.

STUDENT: {name}
PERFORMANCE: {category}
CAREER INTEREST: {interest}

SCORES:
• Attendance: {attendance}% {'(consistent)' if attendance >= 85 else '(needs improvement)' if attendance < 75 else '(moderate)'}
• Math: {math}/100 {'(strong)' if math >= 80 else '(developing)' if math < 60 else '(average)'}
• Programming: {prog}/100 {'(strong)' if prog >= 80 else '(developing)' if prog < 60 else '(average)'}
• Communication: {comm}/100 {'(strong)' if comm >= 80 else '(developing)' if comm < 60 else '(average)'}
• Assignments: {assign}/100 {'(strong)' if assign >= 80 else '(needs focus)' if assign < 60 else '(moderate)'}

STRENGTHS: {strength_summary}
WEAKNESSES: {weakness_summary}

TASK: Generate exactly 3 career paths. Each path MUST:
1. Be specific to THIS student's actual strengths/weaknesses above
2. Include concrete steps with Year 1, Year 2-3 timelines
3. Mention real Indian companies or platforms (TCS, Infosys, Scaler, Coursera, Kaggle, etc.)
4. Be realistic given their {category} performance
5. Be concise (2-3 sentences max per path) - this displays on a dashboard

FORMAT REQUIREMENTS (CRITICAL - FOR DASHBOARD):
• Start each path with **Bold Path Name**
• Keep total path length to 2-3 sentences
• Include specific certifications, companies, or platforms
• Focus on actionable next steps, not generic advice
• DO NOT use flowery language or long paragraphs

EXAMPLE FORMAT (for reference only - customize for {name}):

**Software Development Track**
Your strong programming (85) makes you suited for SDE roles. Year 1: Complete Full Stack course (Udemy/Scaler), build 3 GitHub projects. Year 2-3: Target product companies (Razorpay, Zepto, Cred) or prepare for MAANG with LeetCode Medium consistency.

**Data Analyst Pivot**  
Despite programming gaps (52), your communication strength (78) enables analyst roles. Start with Google Data Analytics cert + SQL mastery, then target Business Analyst positions at TCS Digital or Accenture.

**Service Company Focus**
Build fundamental DSA skills (focus on Easy/Medium problems). Complete Cognizant GenC prep, target mass recruiters (Infosys, Wipro, Capgemini). Safe backup with 4-6 LPA starting package.

NOW GENERATE 3 PATHS FOR {name}. USE THEIR ACTUAL SCORES ({prog} programming, {math} math, {comm} communication). BE SPECIFIC. FOLLOW FORMAT."""

        # ── CALL API (GROQ PREFERRED, FALLBACK TO GEMINI) ──
        if groq_key and GROQ_AVAILABLE:
            roadmaps = _generate_with_groq(prompt, groq_key)
        elif gemini_key:
            roadmaps = _generate_with_gemini(prompt, gemini_key)
        else:
            return _fallback_career_roadmap(student_row, category, careers)
        
        # Ensure exactly 3 paths
        if len(roadmaps) < 3:
            fallback = _fallback_career_roadmap(student_row, category, careers)
            roadmaps += fallback[len(roadmaps):]
        
        return roadmaps[:3]
        
    except Exception as e:
        st.warning(f"AI API call failed, using rule-based roadmap. Error: {e}")
        return _fallback_career_roadmap(student_row, category, careers)


# ── Parent Alert (SMTP-ready stub) ─────────────────────────────────────────────
def build_parent_email(student_row: pd.Series, category: str, careers: str) -> str:
    """Return the email body that would be sent to the parent."""
    name = student_row.get("name", "your ward")
    attendance = student_row.get("attendance", "N/A")

    body = f"""Subject: Early Academic Alert for {name}

Dear Parent/Guardian,

This is an automated early-warning notification from the Academic & Career Navigator.

Our system has identified {name} as currently in the "{category}" performance band.
Key signals:
  • Attendance: {attendance}%
  • Recommended focus areas: {careers or "General Skill Development"}

We strongly recommend a quick conversation at home and coordination with the Student Career Manager,
so that timely support can be provided before any issues become irreversible.

Regards,
AI-Powered Academic & Career Navigator
"""
    return body


# ── Streamlit Layout ───────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="AI-Powered Academic & Career Navigator",
        layout="wide",
    )

    st.title("AI-Powered Academic & Career Navigator")
    st.caption(
        "Hawkathon 2026 · Track 3 · AI-Powered Academic & Career Navigator\n\n"
        "Real-time risk prediction, early alerts, and career roadmaps for every student."
    )

    df = load_students()
    bundle = load_model_bundle()

    if df.empty:
        st.stop()

    tabs = st.tabs(
        [
            "📊 Cohort Overview",
            "🚨 At-Risk Students",
            "🤖 New Student Prediction",
            "📧 Parent Alerts & Roadmaps",
        ]
    )

    # ── Tab 1: Cohort Overview ────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Cohort Performance Snapshot")
        total_students = len(df)
        category_counts = (
            df["performance_category"]
            .value_counts()
            .reindex(ORDERED_LABELS, fill_value=0)
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Students", total_students)
        c2.metric("Top Performers", int(category_counts.get("Top Performer", 0)))
        c3.metric("At-Risk Students", int(category_counts.get("At Risk", 0)))

        st.markdown("#### Distribution by Performance Category")
        st.bar_chart(category_counts)

        st.markdown("#### Sample Student Records")
        st.dataframe(df.head(20), use_container_width=True)

    # ── Tab 2: At-Risk Students ───────────────────────────────────────────────
    with tabs[1]:
        st.subheader("At-Risk Student List (Actionable View)")
        at_risk_df = df[df["performance_category"] == "At Risk"].copy()

        st.write(f"Total At-Risk Students: **{len(at_risk_df)}**")

        display_cols = [
            "name",
            "attendance",
            "math_marks",
            "programming_marks",
            "communication_marks",
            "assignment_score",
            "career_recommendation",
        ]
        display_cols = [c for c in display_cols if c in at_risk_df.columns]

        st.dataframe(
            at_risk_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    # ── Tab 3: New Student Prediction ────────────────────────────────────────
    with tabs[2]:
        st.subheader("Predict Performance & Career Paths for a New Student")

        if bundle is None:
            st.info("Model not available. Please re-run the notebook to generate 'student_model.pkl'.")
        else:
            with st.form("prediction_form"):
                c1, c2, c3 = st.columns(3)
                attendance = c1.number_input("Attendance (%)", min_value=0, max_value=100, value=85)
                math_marks = c2.number_input("Math Marks", min_value=0, max_value=100, value=75)
                programming_marks = c3.number_input(
                    "Programming Marks", min_value=0, max_value=100, value=80
                )

                c4, c5 = st.columns(2)
                communication_marks = c4.number_input(
                    "Communication Marks", min_value=0, max_value=100, value=70
                )
                assignment_score = c5.number_input(
                    "Assignment Score", min_value=0, max_value=100, value=78
                )

                submitted = st.form_submit_button("Predict")

            if submitted:
                result = predict_student_with_model(
                    attendance,
                    math_marks,
                    programming_marks,
                    communication_marks,
                    assignment_score,
                    bundle,
                )
                st.success(
                    f"Predicted Category: **{result['performance_category']}**  \n"
                    f"Career Recommendation: **{result['career_recommendation']}**"
                )

    # ── Tab 4: Parent Alerts & Roadmaps ──────────────────────────────────────
    with tabs[3]:
        st.subheader("Trigger Early Alerts & Generate Career Roadmaps")
        at_risk_df = df[df["performance_category"] == "At Risk"].copy()

        if at_risk_df.empty:
            st.info("No At-Risk students in the current dataset.")
        else:
            names = at_risk_df["name"].tolist()
            selected_name = st.selectbox(
                "Select a student to inspect",
                options=names,
            )
            student_row = at_risk_df[at_risk_df["name"] == selected_name].iloc[0]
            category = student_row.get("performance_category", "At Risk")
            careers = student_row.get("career_recommendation", "General Skill Development")

            st.markdown("##### Student Snapshot")
            st.json(student_row[FEATURE_COLUMNS + ["career_recommendation"]].to_dict())

            st.markdown("##### Parent Alert Email (Preview)")
            email_body = build_parent_email(student_row, category, careers)
            st.code(email_body, language="text")
            st.caption(
                "In a production deployment, this preview would be sent via SMTP "
                "(e.g., Gmail/SendGrid) to the registered parent email."
            )

            st.markdown("##### AI-Powered Career Roadmap (3 Paths)")
            roadmaps = generate_career_roadmap(student_row, category, careers)
            for i, roadmap in enumerate(roadmaps, start=1):
                st.markdown(f"**Path {i}**")
                st.write(roadmap)


if __name__ == "__main__":
    main()

