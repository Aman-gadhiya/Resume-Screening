"""
AI Resume Screening System
---------------------------
A Streamlit application that uses a trained TF-IDF + Linear SVM pipeline
to classify resumes into job categories, run batch screening over many
resumes at once, and explore the model's evaluation results.

Run with:
    streamlit run app.py
"""

import os
import re
import io
import string
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Paths & page config
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [class*="css"]  {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }

    .app-hero {
        background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #C026D3 100%);
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .app-hero h1 {
        font-size: 2.05rem;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }
    .app-hero p {
        font-size: 1.02rem;
        opacity: 0.92;
        margin-bottom: 0;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #ECECF4;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 10px rgba(20, 20, 43, 0.04);
        height: 100%;
    }
    .metric-card .label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6B7280;
        font-weight: 600;
    }
    .metric-card .value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1F2430;
        margin-top: 0.15rem;
    }
    .metric-card .sub {
        font-size: 0.78rem;
        color: #8B8FA3;
        margin-top: 0.15rem;
    }

    .result-card {
        background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
        border: 1px solid #E0E7FF;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card .tag {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6366F1;
        font-weight: 700;
    }
    .result-card .category {
        font-size: 2.1rem;
        font-weight: 800;
        color: #312E81;
        margin: 0.3rem 0;
    }
    .result-card .confidence {
        font-size: 0.95rem;
        color: #4B5563;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1F2430;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid #4F46E5;
        padding-left: 0.6rem;
    }

    .pill {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem;
    }

    .footer-note {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.8rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #EEEEF4;
    }

    div.stButton > button {
        background: linear-gradient(120deg, #4F46E5, #7C3AED);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
        color: white;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    classifier = joblib.load(os.path.join(MODEL_DIR, "resume_classifier.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return vectorizer, classifier, label_encoder


@st.cache_data(show_spinner=False)
def load_reference_data():
    data = {}
    files = {
        "model_comparison": "model_comparison.csv",
        "category_performance": "category_performance.csv",
        "cross_validation": "cross_validation_results.csv",
        "label_mapping": "label_mapping.csv",
        "misclassified": "misclassified_resumes.csv",
    }
    for key, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(path)
            if df.columns[0].startswith("Unnamed"):
                df = df.rename(columns={df.columns[0]: "Category"})
            data[key] = df
        except Exception:
            data[key] = pd.DataFrame()
    return data


@st.cache_data(show_spinner=False)
def load_training_dataset():
    path = os.path.join(DATA_DIR, "UpdatedResumeDataSet.csv")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------------------
# Text cleaning + prediction helpers
# --------------------------------------------------------------------------------------
def clean_resume_text(text: str) -> str:
    """Mirrors the cleaning step used while training the TF-IDF vectorizer."""
    text = re.sub(r"http\S+\s*", " ", text)
    text = re.sub(r"\bRT\b|\bcc\b", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    punctuation_pattern = "[" + re.escape(string.punctuation) + "]"
    text = re.sub(punctuation_pattern, " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def extract_text_from_file(uploaded_file) -> str:
    """Extracts plain text from an uploaded .txt, .pdf, or .docx file.

    Tries multiple libraries for each format so the app still works if only
    one of the optional PDF/DOCX packages is installed in the environment.
    """
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith(".txt"):
        return raw_bytes.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        # Try pypdf, then PyPDF2, then pdfplumber - whichever is installed.
        last_error = None

        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                return text
            last_error = "no extractable text"
        except ImportError:
            last_error = "pypdf not installed"
        except Exception as exc:
            last_error = str(exc)

        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(raw_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                return text
            last_error = "no extractable text"
        except ImportError:
            pass
        except Exception as exc:
            last_error = str(exc)

        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                return text
            last_error = "no extractable text"
        except ImportError:
            pass
        except Exception as exc:
            last_error = str(exc)

        if last_error in ("pypdf not installed",):
            st.error(
                "No PDF reading library is installed in this environment. "
                "Install one and restart the app:\n\n"
                "```bash\npip install pypdf\n```"
            )
        elif last_error == "no extractable text":
            st.warning(
                "No text could be extracted from this PDF - it may be a scanned or "
                "image-based document. Try a text-based PDF, or use **Paste text** instead."
            )
        else:
            st.error(f"Could not read this PDF file: {last_error}")
        return ""

    if name.endswith(".docx"):
        try:
            import docx
            document = docx.Document(io.BytesIO(raw_bytes))
            text = "\n".join(p.text for p in document.paragraphs)
            if text.strip():
                return text
        except ImportError:
            pass
        except Exception as exc:
            st.error(f"Could not read this DOCX file: {exc}")
            return ""

        try:
            import docx2txt
            text = docx2txt.process(io.BytesIO(raw_bytes))
            if text and text.strip():
                return text
        except ImportError:
            st.error(
                "No DOCX reading library is installed in this environment. "
                "Install one and restart the app:\n\n"
                "```bash\npip install python-docx\n```"
            )
            return ""
        except Exception as exc:
            st.error(f"Could not read this DOCX file: {exc}")
            return ""

        st.warning("No text could be extracted from this DOCX file. Try **Paste text** instead.")
        return ""

    st.warning("Unsupported file type. Please upload a .txt, .pdf, or .docx file.")
    return ""


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = scores / temperature
    exp_scores = np.exp(scaled - np.max(scaled))
    return exp_scores / exp_scores.sum()


# LinearSVC's decision-function margins across 25 one-vs-rest classes tend to sit close
# together, which would make a plain softmax look artificially flat (e.g. an obvious,
# correct prediction reading as "6% confidence"). A temperature < 1 sharpens the softmax
# while fully preserving the model's original ranking - it changes how confident the
# score *looks*, never which category wins.
CONFIDENCE_TEMPERATURE = 0.15


def predict_category(raw_text: str, top_k: int = 5):
    vectorizer, classifier, label_encoder = load_pipeline()
    cleaned = clean_resume_text(raw_text)

    if not cleaned or len(cleaned.split()) < 3:
        return None

    features = vectorizer.transform([cleaned])
    prediction = classifier.predict(features)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    scores = classifier.decision_function(features)[0]
    probs = softmax(scores, temperature=CONFIDENCE_TEMPERATURE)

    ranked_idx = np.argsort(probs)[::-1][:top_k]
    ranking = [
        (label_encoder.inverse_transform([i])[0], float(probs[i])) for i in ranked_idx
    ]
    predicted_confidence = float(probs.max())

    return {
        "label": predicted_label,
        "confidence": predicted_confidence,
        "ranking": ranking,
        "cleaned_text": cleaned,
        "word_count": len(cleaned.split()),
    }


# --------------------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.6rem 0 1.1rem 0;">
            <div style="font-size:2.1rem;">📄🤖</div>
            <div style="font-weight:800; font-size:1.15rem; color:#312E81;">
                Resume AI Screener
            </div>
            <div style="font-size:0.8rem; color:#8B8FA3;">
                TF-IDF + Linear SVM · 25 job categories
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "🏠  Home",
            "📂  Batch Screening",
            "📊  Model Insights",
            "ℹ️  About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    try:
        ref = load_reference_data()
        best_model = ref["model_comparison"].sort_values("Accuracy", ascending=False).iloc[0]
        st.markdown("**Deployed model**")
        st.markdown(f"Linear SVM · **{best_model['Accuracy']*100:.2f}%** test accuracy")
    except Exception:
        pass
    st.caption(f"Session: {datetime.now().strftime('%d %b %Y, %H:%M')}")

# --------------------------------------------------------------------------------------
# HOME  (includes the resume classifier front and center)
# --------------------------------------------------------------------------------------
if page.startswith("🏠"):
    st.markdown(
        """
        <div class="app-hero">
            <h1>AI-Powered Resume Screening System</h1>
            <p>Paste or upload a resume below to instantly classify it into the right job
            category, screen entire batches of candidates, and inspect exactly how the
            model performs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Classify a resume (front and center on first view) ----------------
    st.markdown('<div class="section-title">🔍 Classify a resume</div>', unsafe_allow_html=True)
    st.caption("Paste resume text directly, or upload a file. The model predicts the closest matching job category.")

    input_mode = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True, key="home_input_mode")
    resume_text = ""

    if input_mode == "Paste text":
        resume_text = st.text_area(
            "Resume text",
            height=240,
            placeholder="Paste the full resume text here (skills, experience, education, projects)...",
            key="home_resume_text",
        )
    else:
        uploaded = st.file_uploader(
            "Upload a resume file", type=["txt", "pdf", "docx"], key="home_resume_upload"
        )
        if uploaded is not None:
            with st.spinner("Extracting text from file..."):
                resume_text = extract_text_from_file(uploaded)
            if resume_text:
                with st.expander("Preview extracted text"):
                    st.text(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

    predict_clicked = st.button("🔮 Predict category", use_container_width=False, key="home_predict_btn")

    if predict_clicked:
        if not resume_text or not resume_text.strip():
            st.warning("Please provide some resume text before predicting.")
        else:
            with st.spinner("Analyzing resume..."):
                result = predict_category(resume_text)

            if result is None:
                st.warning("The provided text is too short to classify reliably. Please add more content.")
            else:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="tag">Predicted job category</div>
                        <div class="category">{result['label']}</div>
                        <div class="confidence">Model confidence: <b>{result['confidence']*100:.1f}%</b>
                        &nbsp;·&nbsp; {result['word_count']} words analyzed</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_a, col_b = st.columns([1.3, 1])
                with col_a:
                    st.markdown("**Top matching categories**")
                    rank_df = pd.DataFrame(result["ranking"], columns=["Category", "Score"])
                    rank_df["Score"] = rank_df["Score"] * 100
                    fig = px.bar(
                        rank_df.sort_values("Score"),
                        x="Score", y="Category", orientation="h",
                        text=rank_df.sort_values("Score")["Score"].map(lambda v: f"{v:.1f}%"),
                        color="Score", color_continuous_scale="Viridis",
                    )
                    fig.update_layout(
                        height=320, showlegend=False, coloraxis_showscale=False,
                        margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Confidence (%)",
                        yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    st.markdown("**What the model actually reads**")
                    st.caption(
                        "Resume text is lower-cased and stripped of URLs, punctuation, and symbols "
                        "before being converted into TF-IDF word/bigram features — this is the cleaned "
                        "version passed into the model."
                    )
                    st.text_area(
                        "Cleaned text (model input)",
                        value=result["cleaned_text"][:1500],
                        height=230,
                        disabled=True,
                        key="home_cleaned_text",
                    )

    st.markdown("---")

    # ---------------- Headline metrics ----------------
    ref = load_reference_data()
    dataset = load_training_dataset()
    n_categories = dataset["Category"].nunique() if not dataset.empty else 25
    n_resumes = len(dataset) if not dataset.empty else "—"
    try:
        best_row = ref["model_comparison"].sort_values("Accuracy", ascending=False).iloc[0]
        best_acc = f"{best_row['Accuracy']*100:.2f}%"
        best_f1 = f"{best_row['F1 Score']*100:.2f}%"
    except Exception:
        best_acc, best_f1 = "—", "—"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Job Categories", n_categories, "resume classes supported"),
        (c2, "Training Resumes", n_resumes, "labeled training samples"),
        (c3, "Test Accuracy", best_acc, "Linear SVM · held-out test set"),
        (c4, "Weighted F1 Score", best_f1, "precision/recall balance"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            1. **Paste or upload** a resume (`.txt`, `.pdf`, or `.docx`) above.
            2. The text is cleaned with the same preprocessing used during training, then converted
               into TF-IDF features (unigrams + bigrams, 50,000-term vocabulary).
            3. A **Linear Support Vector Machine** predicts the most likely job category and
               ranks the next best alternatives.
            4. Need to screen many candidates at once? Use **Batch Screening** to upload a CSV
               and download categorized results instantly.
            """
        )
        st.markdown('<div class="section-title">Supported categories</div>', unsafe_allow_html=True)
        if not dataset.empty:
            cats = sorted(dataset["Category"].unique())
        else:
            cats = sorted(ref["label_mapping"]["Category"].tolist()) if not ref["label_mapping"].empty else []
        st.markdown(
            "".join(f'<span class="pill">{c}</span>' for c in cats),
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-title">Category distribution (training data)</div>', unsafe_allow_html=True)
        if not dataset.empty:
            counts = dataset["Category"].value_counts().reset_index()
            counts.columns = ["Category", "Count"]
            fig = px.bar(
                counts.sort_values("Count"),
                x="Count", y="Category",
                orientation="h",
                color="Count",
                color_continuous_scale="Purples",
            )
            fig.update_layout(
                height=560, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Training dataset not found in `data/`.")

# --------------------------------------------------------------------------------------
# BATCH SCREENING
# --------------------------------------------------------------------------------------
elif page.startswith("📂"):
    st.markdown('<div class="section-title">Batch resume screening</div>', unsafe_allow_html=True)
    st.caption(
        "Upload a CSV file containing a column of resume text to classify many resumes at once. "
        "A file with columns like the training set (`Category`, `Resume`) also works — the "
        "`Category` column will simply be ignored for prediction."
    )

    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_csv is not None:
        try:
            batch_df = pd.read_csv(uploaded_csv)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            batch_df = None

        if batch_df is not None and not batch_df.empty:
            st.write(f"Loaded **{len(batch_df)}** rows.")
            text_col = st.selectbox(
                "Which column contains the resume text?",
                options=batch_df.columns.tolist(),
                index=(batch_df.columns.tolist().index("Resume") if "Resume" in batch_df.columns else 0),
            )

            if st.button("🚀 Run batch prediction", use_container_width=False):
                vectorizer, classifier, label_encoder = load_pipeline()
                progress = st.progress(0, text="Classifying resumes...")

                texts = batch_df[text_col].astype(str).tolist()
                cleaned_texts = [clean_resume_text(t) for t in texts]
                features = vectorizer.transform(cleaned_texts)
                preds = classifier.predict(features)
                labels = label_encoder.inverse_transform(preds)

                scores = classifier.decision_function(features)
                confidences = np.array(
                    [softmax(row, temperature=CONFIDENCE_TEMPERATURE).max() for row in scores]
                )
                progress.progress(100, text="Done")

                result_df = batch_df.copy()
                result_df["Predicted_Category"] = labels
                result_df["Confidence_%"] = (confidences * 100).round(2)

                st.success(f"Classified {len(result_df)} resumes.")
                st.dataframe(result_df, use_container_width=True, height=380)

                col1, col2 = st.columns([1, 1])
                with col1:
                    dist = result_df["Predicted_Category"].value_counts().reset_index()
                    dist.columns = ["Category", "Count"]
                    fig = px.pie(dist, names="Category", values="Count", hole=0.45)
                    fig.update_layout(
                        height=420, margin=dict(l=10, r=10, t=30, b=10),
                        title="Predicted category breakdown",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig2 = px.histogram(result_df, x="Confidence_%", nbins=20, color_discrete_sequence=["#7C3AED"])
                    fig2.update_layout(
                        height=420, margin=dict(l=10, r=10, t=30, b=10),
                        title="Confidence score distribution", xaxis_title="Confidence (%)", yaxis_title="Resumes",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions as CSV",
                    data=csv_bytes,
                    file_name=f"resume_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
    else:
        st.info("Upload a CSV to begin. You can also try the shipped `UpdatedResumeDataSet.csv` sample.")

# --------------------------------------------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------------------------------------------
elif page.startswith("📊"):
    st.markdown('<div class="section-title">Model performance & evaluation</div>', unsafe_allow_html=True)
    ref = load_reference_data()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Model comparison", "Category performance", "Confusion matrix", "Cross-validation & errors"]
    )

    with tab1:
        mc = ref["model_comparison"]
        if not mc.empty:
            st.dataframe(
                mc.style.format({c: "{:.2%}" for c in mc.columns if c != "Model"}),
                use_container_width=True,
            )
            melted = mc.melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig = px.bar(
                melted, x="Model", y="Score", color="Metric", barmode="group",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(
                yaxis_tickformat=".0%", height=420, margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("`model_comparison.csv` not found.")

    with tab2:
        cp = ref["category_performance"]
        if not cp.empty:
            st.dataframe(cp, use_container_width=True, height=420)
            metric_cols = [c for c in ["precision", "recall", "f1-score"] if c in cp.columns]
            if metric_cols:
                melted = cp.melt(id_vars="Category", value_vars=metric_cols, var_name="Metric", value_name="Score")
                fig = px.bar(
                    melted, x="Category", y="Score", color="Metric", barmode="group",
                    color_discrete_sequence=["#4F46E5", "#7C3AED", "#EC4899"],
                )
                fig.update_layout(height=460, xaxis_tickangle=-45, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("`category_performance.csv` not found.")

    with tab3:
        cm_path = os.path.join(DATA_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True, caption="Confusion matrix — final deployed model")
        else:
            st.info("`confusion_matrix.png` not found.")

    with tab4:
        cv = ref["cross_validation"]
        mis = ref["misclassified"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cross-validation results**")
            if not cv.empty:
                st.dataframe(cv, use_container_width=True, height=250)
            else:
                st.info("`cross_validation_results.csv` not found.")
        with c2:
            st.markdown("**Sample misclassified resumes**")
            if not mis.empty:
                st.dataframe(mis, use_container_width=True, height=250)
            else:
                st.info("`misclassified_resumes.csv` not found.")

# --------------------------------------------------------------------------------------
# ABOUT
# --------------------------------------------------------------------------------------
else:
    st.markdown('<div class="section-title">About this project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This application wraps a trained **resume screening classifier** in an interactive,
        production-style interface.

        **Pipeline**
        - Text cleaning (URL/punctuation/symbol removal, lower-casing)
        - **TF-IDF** vectorization — unigrams & bigrams, up to 50,000 terms
        - **Linear Support Vector Machine** classifier (`class_weight="balanced"`)
        - Label encoding across **25 job categories**

        **What each page does**
        - **Home** — classify a resume immediately (paste or upload), plus headline metrics
          and the category distribution of the training data
        - **Batch Screening** — classify an entire CSV of resumes and export the results
        - **Model Insights** — model comparison, per-category precision/recall/F1, the confusion
          matrix, cross-validation results, and sample misclassifications

        **Note on confidence scores.** The underlying model is a Linear SVM, which does not output
        true probabilities. Confidence values shown in this app are derived by applying a softmax
        to the model's decision-function scores, giving a normalized, easy-to-read relative ranking
        across categories — not a calibrated probability.
        """
    )
    st.markdown('<div class="section-title">Disclaimer</div>', unsafe_allow_html=True)
    st.caption(
        "This tool is intended to assist, not replace, human judgement in recruitment. "
        "Predictions should be reviewed by a human before any hiring decision is made."
    )

st.markdown(
    '<div class="footer-note">AI Resume Screening System · Built with Streamlit, scikit-learn & Plotly</div>',
    unsafe_allow_html=True,
)
