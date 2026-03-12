# AI-Powered-Academic-Career-Navigator

AI-powered platform to help institutions detect at-risk students early, notify parents proactively, and generate personalized career roadmaps using student performance data.

## 1. Problem & Motivation

Current institutional workflows rely on fragmented spreadsheets and late, manual reporting. By the time a student is flagged as “at-risk”, a full semester is often lost, parents find out too late, and career guidance remains generic instead of personalized.

This project implements the **Hawkathon 2026 – EdTech Track 3: AI-Powered Academic & Career Navigator** by:

- **Centralizing data**: using a unified dataset of attendance and marks as a proxy for LMS data.
- **Predicting risk**: clustering students into `Top Performer`, `Average`, and `At Risk` using K-Means.
- **Automating alerts**: generating SMTP-ready parent alert emails the moment a student falls into the `At Risk` band.
- **Personalizing careers**: producing structured, AI-style career roadmaps tailored to each student’s strengths.
- **Empowering SCMs**: providing a simple dashboard where Student Career Managers can see the entire cohort and drill into at-risk profiles.

## 2. Solution Overview

At a high level, the system consists of:

- **Data Layer**
  - Input: `students_dataset.csv` / `students_clustered.csv` (attendance, subject marks, assignment scores, interests).
  - Unified, live-queryable dataset (in a full deployment this maps to PostgreSQL/MongoDB).

- **ML Layer (Risk Prediction)**
  - Standardizes key academic features (attendance + marks).
  - Trains a **K-Means** model with 3 clusters.
  - Ranks clusters by mean performance and maps them to: `At Risk`, `Average`, `Top Performer`.
  - Persists the trained `scaler`, `kmeans`, and `cluster_label_map` in `student_model.pkl`.

- **Analytics & Dashboard Layer**
  - A **Streamlit** app (`app.py`) that shows:
    - Cohort overview (total students, category distribution).
    - A focused table of **At-Risk** students with their scores and recommendations.
    - A form to run **new student predictions** in real time using the saved model.

- **Communication Layer (Parent Alerts)**
  - Builds a clear, human-readable parent email for any at-risk student.
  - Currently exposes a **preview** of the email body; in production this is wired to SMTP (e.g. Gmail/SendGrid).

- **Career Guidance Layer (Roadmaps)**
  - For each student, generates **3 structured career paths** based on:
    - Performance category (`Top Performer` / `Average` / `At Risk`).
    - Strength signals (high programming, maths, communication scores, attendance).
    - Stated `career_interest` where available.
  - Designed so it can be upgraded easily to call Gemini/OpenAI APIs directly.

## 3. Tech Stack

- **Language**: Python
- **ML**: scikit-learn (K-Means, StandardScaler)
- **Data**: pandas, numpy
- **Dashboard**: Streamlit
- **Visualization**: matplotlib, seaborn (used in notebook)

## 4. Project Structure (Key Files)

- `Main_improved.ipynb` – end-to-end notebook:
  - Loads and validates the dataset.
  - Performs feature scaling and trains K-Means.
  - Maps raw clusters → `performance_category`.
  - Generates `career_recommendation` strings.
  - Saves `students_clustered.csv` and `student_model.pkl`.
  - Provides a `predict_student(...)` helper for quick testing.

- `app.py` – Streamlit SCM dashboard:
  - Loads `students_clustered.csv` and `student_model.pkl`.
  - Shows cohort summary and performance distribution.
  - Lists all `At Risk` students in an actionable table.
  - Offers a form to predict performance + careers for a **new** student.
  - Previews parent alerts and generates 3-path career roadmaps.

- `students_dataset.csv` – base synthetic dataset (input).
- `students_clustered.csv` – enriched dataset with model outputs (`cluster`, `performance_category`, `career_recommendation`).
- `student_model.pkl` – persisted ML bundle (StandardScaler + KMeans + cluster → label mapping).
- `requirements.txt` – minimal dependencies to run the notebook and dashboard.

## 5. How to Run Locally

### 5.1. Setup Environment

```bash
cd "AI-Powered-Academic-Career-Navigator"
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install -r requirements.txt
```

> Note: The notebook also uses matplotlib/seaborn for visual exploration; these are already included in `requirements.txt`.

### 5.2. Generate Model & Clustered Data (if needed)

If `students_clustered.csv` or `student_model.pkl` are missing or outdated:

1. Open `Main_improved.ipynb` in Jupyter / VS Code / Cursor.
2. Run all cells from top to bottom.
3. Confirm that:
   - `students_clustered.csv` is created.
   - `student_model.pkl` is created.

### 5.3. Launch the SCM Dashboard

```bash
streamlit run app.py
```

Then open the provided local URL in your browser. You will see:

- **Cohort Overview** tab – total students, category counts, and a bar chart.
- **At-Risk Students** tab – list of students that require immediate attention.
- **New Student Prediction** tab – quick inference for a single student profile.
- **Parent Alerts & Roadmaps** tab – email preview + 3 career paths for any at-risk student.

## 6. Mapping to Hawkathon 2026 Track 3 Requirements

- **Centralize Data (Unified LMS Database)**
  - This prototype uses CSV files as a stand-in for a production PostgreSQL/MongoDB backend.
  - The schema already mirrors typical LMS fields (attendance, subject marks, assignments, interests).

- **Predict Risk (Unsupervised ML – K-Means)**
  - K-Means clustering segments the cohort into three tiers.
  - Clusters are mapped to labels based on average performance instead of hand-written thresholds, reducing bias.

- **Automate Communication (SMTP Alerts)**
  - The dashboard can build a ready-to-send email body for any at-risk student.
  - In a real deployment, this connects to SMTP providers to send live alerts to registered parent emails.

- **Personalize Careers (Generative AI Roadmaps)**
  - For each student, the app generates 3 possible career paths backed by their strengths and category.
  - The `generate_career_roadmap` function is designed to be upgraded to Gemini/OpenAI with minimal changes.

- **SCM Command & Control Dashboard**
  - Streamlit UI gives SCMs a real-time snapshot of the entire cohort and allows drilling down into the most critical cases.

## 7. Future Work

- Integrate with a real LMS and database (PostgreSQL/MongoDB) instead of CSV.
- Wire the parent alert preview to production-ready SMTP (SendGrid/Gmail) with templates and analytics.
- Replace rule-based roadmaps with live Gemini/OpenAI calls, including explanation fields.
- Add authentication and role-based access control (SCM vs faculty vs admin).
- Extend the feature space (extracurriculars, club activity, prior semesters) and experiment with more advanced clustering / anomaly detection.
