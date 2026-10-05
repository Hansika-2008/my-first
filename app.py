import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------
# AI RESUME SCREENING SYSTEM
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("🤖 AI Resume Screening System")
st.write("Upload resumes and compare them with a job description using AI-based text similarity.")


# ---------------------------------------------------
# FUNCTION: Extract text from PDF
# ---------------------------------------------------

def extract_text_from_pdf(uploaded_file):
    text = ""

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        st.error(f"Could not read PDF: {e}")

    return text


# ---------------------------------------------------
# FUNCTION: Clean text
# ---------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


# ---------------------------------------------------
# FUNCTION: Calculate similarity
# ---------------------------------------------------

def calculate_score(job_description, resume_text):

    job_description = clean_text(job_description)
    resume_text = clean_text(resume_text)

    documents = [job_description, resume_text]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)


# ---------------------------------------------------
# FUNCTION: Find matching skills
# ---------------------------------------------------

def find_skills(job_description, resume_text):

    skills = [
        "python",
        "java",
        "c",
        "c++",
        "javascript",
        "html",
        "css",
        "sql",
        "mysql",
        "mongodb",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "data science",
        "data analysis",
        "pandas",
        "numpy",
        "scikit learn",
        "tensorflow",
        "keras",
        "django",
        "flask",
        "react",
        "node js",
        "git",
        "github",
        "cloud",
        "aws",
        "azure",
        "power bi"
    ]

    job_text = clean_text(job_description)
    resume_text = clean_text(resume_text)

    required_skills = []
    matched_skills = []

    for skill in skills:

        if skill in job_text:
            required_skills.append(skill)

            if skill in resume_text:
                matched_skills.append(skill)

    return required_skills, matched_skills


# ---------------------------------------------------
# FUNCTION: Screening result
# ---------------------------------------------------

def get_result(score):

    if score >= 75:
        return "Excellent Match", "🟢"

    elif score >= 60:
        return "Good Match", "🟡"

    elif score >= 40:
        return "Average Match", "🟠"

    else:
        return "Low Match", "🔴"


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("⚙️ Resume Screening")

st.sidebar.write(
    "This system uses TF-IDF and cosine similarity "
    "to compare resumes with the job description."
)


# ---------------------------------------------------
# JOB DESCRIPTION
# ---------------------------------------------------

st.header("1️⃣ Job Description")

job_description = st.text_area(
    "Enter the job description:",
    height=200,
    placeholder=(
        "Example:\n"
        "We are looking for a Python developer with knowledge "
        "of SQL, Machine Learning, Pandas and NumPy."
    )
)


# ---------------------------------------------------
# RESUME UPLOAD
# ---------------------------------------------------

st.header("2️⃣ Upload Resumes")

uploaded_files = st.file_uploader(
    "Upload one or more PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)


# ---------------------------------------------------
# SCREEN RESUMES
# ---------------------------------------------------

if st.button("🚀 Screen Resumes"):

    if not job_description.strip():

        st.warning("Please enter a job description.")

    elif not uploaded_files:

        st.warning("Please upload at least one resume.")

    else:

        results = []

        with st.spinner("Analyzing resumes..."):

            for uploaded_file in uploaded_files:

                resume_text = extract_text_from_pdf(
                    uploaded_file
                )

                if not resume_text.strip():

                    continue

                score = calculate_score(
                    job_description,
                    resume_text
                )

                required_skills, matched_skills = find_skills(
                    job_description,
                    resume_text
                )

                result, icon = get_result(score)

                if required_skills:

                    skill_percentage = round(
                        len(matched_skills)
                        / len(required_skills)
                        * 100,
                        2
                    )

                else:

                    skill_percentage = 0

                results.append({
                    "Resume": uploaded_file.name,
                    "AI Score": score,
                    "Result": result,
                    "Required Skills": len(required_skills),
                    "Matched Skills": len(matched_skills),
                    "Skill Match": skill_percentage,
                    "Skills Found": ", ".join(matched_skills)
                })


        # ---------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------

        if results:

            st.header("3️⃣ Screening Results")

            # Sort by score
            results = sorted(
                results,
                key=lambda x: x["AI Score"],
                reverse=True
            )

            for rank, result in enumerate(results, start=1):

                st.subheader(
                    f"#{rank} {result['Resume']}"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "AI Match Score",
                        f"{result['AI Score']}%"
                    )

                with col2:
                    st.metric(
                        "Skill Match",
                        f"{result['Skill Match']}%"
                    )

                with col3:
                    st.metric(
                        "Matched Skills",
                        result["Matched Skills"]
                    )

                with col4:
                    st.write(
                        f"**{result['Result']}**"
                    )

                if result["Skills Found"]:

                    st.success(
                        "Skills Found: "
                        + result["Skills Found"]
                    )

                else:

                    st.info(
                        "No specific skills matched."
                    )

                st.divider()


            # ---------------------------------------------------
            # RANKING TABLE
            # ---------------------------------------------------

            st.header("📊 Candidate Ranking")

            table_data = []

            for rank, result in enumerate(
                results,
                start=1
            ):

                table_data.append({
                    "Rank": rank,
                    "Resume": result["Resume"],
                    "AI Score (%)": result["AI Score"],
                    "Skill Match (%)": result["Skill Match"],
                    "Matched Skills": result["Matched Skills"],
                    "Result": result["Result"]
                })

            st.dataframe(
                table_data,
                use_container_width=True
            )


            # ---------------------------------------------------
            # BEST CANDIDATE
            # ---------------------------------------------------

            best_candidate = results[0]

            st.header("🏆 Best Matching Candidate")

            st.success(
                f"**{best_candidate['Resume']}** "
                f"is the best match with an AI score of "
                f"**{best_candidate['AI Score']}%**."
            )

        else:

            st.error(
                "Could not extract text from the uploaded resumes."
            )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "AI Resume Screening System | Python + Streamlit + "
    "TF-IDF + Cosine Similarity"
)