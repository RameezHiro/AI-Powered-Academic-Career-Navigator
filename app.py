import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai


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
    """Deterministic roadmap used when Gemini is not configured/reachable."""
    strengths = []
    if student_row.get("programming_marks", 0) >= 80:
        strengths.append("strong programming fundamentals")
    if student_row.get("math_marks", 0) >= 80:
        strengths.append("solid mathematical / analytical skills")
    if student_row.get("communication_marks", 0) >= 80:
        strengths.append("excellent communication & presentation")
    if student_row.get("attendance", 0) >= 85:
        strengths.append("consistent discipline & attendance")

    strengths_text = ", ".join(strengths) or "a mix of foundational technical and soft skills"
    interest = student_row.get("career_interest", "general technology and business roles")

    base = f"Based on {category} performance and {strengths_text}, "
    primary_track = careers.split(",")[0].strip() if careers else interest

    roadmap_1 = (
        base
        + f"focus on a 2–3 year path into {primary_track}: "
        "deepen core subjects, complete 2–3 major projects, and aim for internships aligned with this track."
    )
    roadmap_2 = (
        "Develop a parallel skill stack: combine your current degree with complementary online "
        "certifications (Coursera, NPTEL, etc.), contribute to GitHub or college clubs, and build a "
        "portfolio that proves your skills to recruiters."
    )
    roadmap_3 = (
        "By final year, target 3 categories of companies (product, service, startups), prepare a "
        "structured interview plan (DSA + role knowledge), and use alumni/LinkedIn outreach "
        "to secure 5–10 warm referrals."
    )

    return [roadmap_1, roadmap_2, roadmap_3]


def generate_career_roadmap(student_row: pd.Series, category: str, careers: str) -> list[str]:
    """
    Generate 3 roadmap suggestions for a student using Gemini if available.

    Falls back to a deterministic heuristic version if GEMINI_API_KEY is not set
    or if the API call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_career_roadmap(student_row, category, careers)

    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "You are an expert academic and career counselor for Indian undergraduate students. "
            "Given the following student profile, generate THREE distinct 2–3 year career roadmaps. "
            "Each roadmap should be a short paragraph, practical, and tailored to this student.\n\n"
            f"Student name: {student_row.get('name', 'N/A')}\n"
            f"Performance category: {category}\n"
            f"Attendance: {student_row.get('attendance', 'N/A')}%\n"
            f"Math marks: {student_row.get('math_marks', 'N/A')}\n"
            f"Programming marks: {student_row.get('programming_marks', 'N/A')}\n"
            f"Communication marks: {student_row.get('communication_marks', 'N/A')}\n"
            f"Assignment score: {student_row.get('assignment_score', 'N/A')}\n"
            f"Career interest (if any): {student_row.get('career_interest', 'N/A')}\n"
            f"Model-derived recommended careers: {careers or 'General Skill Development'}\n\n"
            "Return the answer as three numbered points 1., 2., and 3., without any extra commentary."
        )

        response = model.generate_content(prompt)
        text = response.text or ""

        # Split into 3 points based on numbering; be defensive.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        roadmaps: list[str] = []
        current = []
        for line in lines:
            if line.startswith(("1.", "2.", "3.")) and current:
                roadmaps.append(" ".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            roadmaps.append(" ".join(current).strip())

        # Ensure exactly 3 items; pad/trim using fallback if needed.
        if len(roadmaps) < 3:
            fallback = _fallback_career_roadmap(student_row, category, careers)
            roadmaps += fallback[len(roadmaps) :]
        elif len(roadmaps) > 3:
            roadmaps = roadmaps[:3]

        return roadmaps
    except Exception as e:
        # Show a gentle warning in the UI and fall back.
        st.warning(f"Gemini API call failed, using heuristic roadmap instead. ({e})")
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

