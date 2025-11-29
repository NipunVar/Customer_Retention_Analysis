# streamlit_app.py
"""
Streamlit app for Customer Retention project.

- Scans current working directory for media, notebooks, scripts, CSVs, README.md
- Shows fixed pages: Overview, Exploratory Images, Feature Engineering, Model Results, Prediction Explorer
- Adds dynamic folder pages for the folder names you provided:
  - Deep Learning with Tensorflow
  - ETL process
  - Exploratory Data Analysis
  - Feature_generation
  - Model fit & CV
- Each folder page lists only files that belong to that folder (robust matching).
- If a folder has no Python scripts, the "Python Script" section is omitted entirely.
- Improved image UI: large preview scales to available column width (no horizontal overflow).
- Prediction Explorer: nicer two-column layout, filters, metrics, charts, download.
- Overview uses your requested title, contact block and formatted project content.
"""

import os
import re
import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Customer Retention Project", layout="wide")

# -----------------------------------------------------
# Constants
# -----------------------------------------------------
ROOT = os.getcwd()
SUFFIXES = [
    ".png", ".jpg", ".jpeg", ".gif", ".pptx",
    ".ipynb", ".py", ".csv", "README.md"
]

# Folder titles to expose as dynamic sections
DL_FOLDER = "Deep Learning with Tensorflow"
ETL_FOLDER = "ETL process"
EDA_FOLDER = "Exploratory Data Analysis"
FEATURE_FOLDER = "Feature_generation"
MODEL_FOLDER = "Model fit & CV"

FOLDER_LIST = [DL_FOLDER, ETL_FOLDER, EDA_FOLDER, FEATURE_FOLDER, MODEL_FOLDER]

# -----------------------------------------------------
# File scanner
# -----------------------------------------------------
def scan_recursive(root_dir):
    found = {s: [] for s in SUFFIXES}
    for base, _, files in os.walk(root_dir):
        for f in files:
            fl = f.lower()
            for s in SUFFIXES:
                if s == "README.md" and fl == "readme.md":
                    found[s].append(os.path.join(base, f))
                elif fl.endswith(s):
                    found[s].append(os.path.join(base, f))
    return found

found = scan_recursive(ROOT)

def list_images(found_dict):
    imgs = []
    for ext in (".png", ".jpg", ".jpeg", ".gif"):
        imgs.extend(found_dict.get(ext, []))
    return sorted(imgs)

def show_thumbnail(path, width=None, caption=None):
    """
    Show a thumbnail using st.image; width controls size.
    """
    try:
        img = Image.open(path).convert("RGB")
        st.image(img, caption=caption or os.path.basename(path), use_container_width=(width is None), width=width)
    except Exception as e:
        st.error(f"Could not load {path}: {e}")

def show_large_centered(path, caption=None):
    """
    Display a larger image centered using columns and use_container_width so image scales responsively.
    """
    try:
        img = Image.open(path).convert("RGB")
        left, center, right = st.columns([1, 3, 1])  # center column wider
        with center:
            st.image(img, caption=caption or os.path.basename(path), use_container_width=True)
    except Exception as e:
        st.error(f"Could not load {path}: {e}")

def load_readme(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def pick_csv(paths):
    for p in sorted(paths):
        if "prediction" in os.path.basename(p).lower():
            return p
    return None

images = list_images(found)
csvs = found.get(".csv", [])
readme_files = found.get("README.md", [])

all_notebooks = sorted(found.get(".ipynb", []))
all_scripts = sorted(found.get(".py", []))

# -----------------------------------------------------
# Robust folder-membership scoring & single assignment
# - Each file is scored against every folder and assigned to the best-scoring folder.
# -----------------------------------------------------
def make_acronym(s):
    parts = re.findall(r"[A-Za-z0-9]+", s)
    if not parts:
        return ""
    return "".join(p[0].lower() for p in parts)

def normalize_tokens(s):
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", s)]

def score_match(path, folder_name):
    norm_path = os.path.normpath(path)
    parts = [p for p in norm_path.split(os.sep) if p]
    parent = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)
    base_no_ext = os.path.splitext(filename)[0].lower()

    score = 0
    if parent == folder_name:
        score += 50
    if folder_name in parts:
        score += 30
    if base_no_ext == folder_name.lower():
        score += 40

    folder_tokens = normalize_tokens(folder_name)
    file_tokens = normalize_tokens(filename)
    path_tokens = []
    for p in parts:
        path_tokens.extend(normalize_tokens(p))
    token_matches = sum(1 for t in folder_tokens if t in file_tokens or t in path_tokens)
    score += 5 * token_matches

    acronym = make_acronym(folder_name)
    if acronym and acronym in filename.lower():
        score += 20

    return score

def assign_files_to_folders(file_paths, folder_list):
    mapping = {f: [] for f in folder_list}
    for p in file_paths:
        best_score = 0
        best_folders = []
        for f in folder_list:
            s = score_match(p, f)
            if s > best_score:
                best_score = s
                best_folders = [f]
            elif s == best_score and s > 0:
                best_folders.append(f)
        if best_score == 0:
            continue
        if len(best_folders) > 1:
            parent = os.path.basename(os.path.dirname(p))
            parent_matches = [f for f in best_folders if parent == f]
            if parent_matches:
                chosen = parent_matches[0]
            else:
                for f in folder_list:
                    if f in best_folders:
                        chosen = f
                        break
        else:
            chosen = best_folders[0]
        mapping[chosen].append(p)
    return mapping

notebook_mapping = assign_files_to_folders(all_notebooks, FOLDER_LIST)
script_mapping = assign_files_to_folders(all_scripts, FOLDER_LIST)

dl_notebooks = notebook_mapping.get(DL_FOLDER, [])
dl_scripts = script_mapping.get(DL_FOLDER, [])

etl_notebooks = notebook_mapping.get(ETL_FOLDER, [])
etl_scripts = script_mapping.get(ETL_FOLDER, [])

eda_notebooks = notebook_mapping.get(EDA_FOLDER, [])
eda_scripts = script_mapping.get(EDA_FOLDER, [])

feature_notebooks = notebook_mapping.get(FEATURE_FOLDER, [])
feature_scripts = script_mapping.get(FEATURE_FOLDER, [])

model_notebooks = notebook_mapping.get(MODEL_FOLDER, [])
model_scripts = script_mapping.get(MODEL_FOLDER, [])

# -----------------------------------------------------
# Sidebar & page selection (reordered per your request)
# -----------------------------------------------------
PAGE_ORDER = [
    "Overview",
    "Prediction Explorer",
    DL_FOLDER,
    ETL_FOLDER,
    EDA_FOLDER,
    FEATURE_FOLDER,
    MODEL_FOLDER,
    "Exploratory Images",
    "Feature Engineering",
    "Model Results",
]

st.sidebar.title("Project Navigator")
page = st.sidebar.radio("Go to", PAGE_ORDER)

# -----------------------------------------------------
# Helpers for images lists & rendering
# -----------------------------------------------------
def images_for_exploratory():
    return images

def images_for_feature_engineering():
    return [p for p in images if "feature" in os.path.basename(p).lower()]

def images_for_model_results():
    perf = [p for p in images if "perf" in os.path.basename(p).lower() or "performance" in os.path.basename(p).lower()]
    imp = [p for p in images if "importance" in os.path.basename(p).lower()]
    seen = set()
    out = []
    for p in perf + imp:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def render_image_selector_page(title, image_list_fn):
    st.title(title)
    imgs = image_list_fn()

    if not imgs:
        st.info("No images found for this section.")
        return

    labels = [os.path.basename(p) for p in imgs]

    sel = st.selectbox("Choose image to preview", ["-- Select an image --"] + labels)
    if sel != "-- Select an image --":
        idx = labels.index(sel)
        selected_path = imgs[idx]
        # show large centered responsive image (fits container)
        show_large_centered(selected_path, caption=sel)

    show_grid = st.checkbox("Show gallery thumbnails", value=False)
    if show_grid:
        st.markdown("---")
        st.markdown("#### Gallery")
        cols_per_row = 4
        for i in range(0, len(imgs), cols_per_row):
            row_imgs = imgs[i:i+cols_per_row]
            cols = st.columns(len(row_imgs))
            for col, img_path in zip(cols, row_imgs):
                with col:
                    show_thumbnail(img_path, width=220, caption=os.path.basename(img_path))
                    st.caption("")

# -----------------------------------------------------
# Overview content (formatted visually)
# Title -> Contact block -> Formatted content
# -----------------------------------------------------
OVERVIEW_TITLE = "E-commerce Repeat Buyers Churn Prediction Using Machine Learning"

OVERVIEW_CONTACT = {
    "name": "Nipun Varshneya",
    "email": "varshneya.nipun16@gmail.com",
    "phone": "+91 7303615333",
    "github": "https://github.com/NipunVar/Customer_Retention_Analysis"
}

OVERVIEW_CONTENT = {
    "Description": (
        "Merchants often run large promotions (e.g., discounts or cash coupons) on major sales events "
        "like Boxing Day, Black Friday, or Double 11 (Nov 11th) to attract new buyers. Many of these buyers "
        "are one-time deal hunters, limiting the long-term impact on sales. Predicting which customers are likely "
        "to become repeat buyers helps merchants target potential loyal customers, reduce promotion costs, and improve ROI."
    ),
    "Data": (
        "The dataset is provided by Alibaba Cloud and contains anonymized user shopping logs from the six months "
        "leading up to \"Double 11\" along with labels indicating repeat buyers. Due to privacy concerns, the data "
        "is sampled, which may not reflect the exact statistics of Tmall.com, but it is sufficient for modeling purposes."
    ),
    "Data Processing": (
        "The training dataset contains 260,864 users' data, including profile information and user activity logs. "
        "Data was processed in chunks (pandas chunksize=10000) to manage memory efficiently."
    ),
    "Exploratory Data Analysis (EDA)": (
        "- User Profile: demographics and profile information\n"
        "- User Behavior: purchase and browsing patterns\n"
        "- Total Actions: overall user activity counts\n"
        "- Action by Month: monthly distribution of actions"
    ),
    "Feature Engineering": (
        "Features were generated using aggregation and grouping techniques. The final dataset includes 81 features "
        "grouped into:\n\n"
        "- Action-Based Features\n- Day-Based Features\n- Product Diversity\n- User-Merchant Similarity\n- Recent Activities"
    ),
    "Model Fitting": (
        "Traditional models evaluated: Random Forest, Logistic Regression, Gradient Boosting Machine, XGBoost. "
        "Sampling techniques used to address imbalance: SMOTE, Random Under Sampler, ADASYN. "
        "A stratified k-fold cross-validation procedure was used to evaluate models with multiple scoring metrics."
    ),
    "Results": (
        "Best Performing Model: XGBoost with SMOTE. The model produced the strongest accuracy and competitive AUC. "
        "Feature importance analysis helped identify the most predictive variables for repeat-buyer behavior."
    ),
    "Deep Learning Extension": (
        "A TensorFlow/Keras neural network was implemented to further boost performance: added a hidden layer, "
        "applied class-weighting and oversampling strategies, and monitored validation metrics to avoid overfitting."
    ),
    "References": (
        "- Guimei Liu, Tam T. Nguyen, Gang Zhao. Repeat Buyer Prediction for E-Commerce. KDD 2016\n"
        "- Rahul Bhagat, Srevatsan Muralidharan. Buy It Again: Modeling Repeat Purchase Recommendations. KDD 2018\n"
        "- Huibing Zhang, Junchao Dong. Prediction of Repeat Customers... Wireless Communications and Mobile Computing, 2020\n"
        "- D. M. Blei, A. Y. Ng, M. I. Jordan. Latent Dirichlet Allocation. JMLR, 2003\n"
        "- L. Breiman. Random Forests. Machine Learning, 2001\n"
        "- T. Chen, T. He. XGBoost: Extreme Gradient Boosting. GitHub\n"
    )
}

# -----------------------------------------------------
# PAGE: Overview (updated layout)
# -----------------------------------------------------
if page == "Overview":
    # Title (prominent)
    st.markdown(f"# {OVERVIEW_TITLE}")

    # Contact block directly beneath title (left column small, right column link)
    name_col, contact_col = st.columns([1, 2])
    with name_col:
        st.markdown(f"**{OVERVIEW_CONTACT['name']}**")
        st.write(OVERVIEW_CONTACT["email"])
        st.write(OVERVIEW_CONTACT["phone"])
    with contact_col:
        st.markdown(f"**GitHub Repository**")
        st.write(OVERVIEW_CONTACT["github"])

    st.markdown("---")

    # Render content sections visually: header + paragraph / bullets
    for section, text in OVERVIEW_CONTENT.items():
        st.subheader(section)
        # For sections that include newline-separated bullets, render as markdown so bullets format correctly
        if "\n-" in text or text.strip().startswith("-"):
            st.markdown(text)
        else:
            # keep line breaks with markdown
            st.markdown(text)
        st.markdown("")  # small space

# -----------------------------------------------------
# PAGE: Prediction Explorer (enhanced UI)
# - visually improved: overview, metrics, tabs for preview/charts/summary, better filters
# -----------------------------------------------------
elif page == "Prediction Explorer":
    st.title("Prediction Explorer")

    st.markdown(
        """
        **About this explorer:** This interactive Prediction Explorer lets you load the model prediction CSV, quickly
        inspect samples, apply filters and text searches, and compute basic metrics (e.g., predicted positive rate).
        Use the left-hand controls to slice the dataset; the right-hand area provides a responsive preview, summary
        charts, and download of the filtered subset. The goal is to make it easy to validate model outputs, inspect
        edge cases, and export ready-to-share CSVs for downstream analysis or reporting.
        """
    )

    pred_csv = pick_csv(csvs)
    if not pred_csv:
        st.info("No prediction CSV found (looking for filenames containing 'prediction').")
    else:
        st.success(f"Loaded: {os.path.basename(pred_csv)}")
        try:
            df = pd.read_csv(pred_csv)
        except Exception:
            df = pd.read_csv(pred_csv, encoding="latin-1")

        lower_cols = [c.lower() for c in df.columns]
        prediction_candidates = [c for c in df.columns if any(k in c.lower() for k in ("pred", "score", "prob", "probability", "prediction"))]
        label_candidates = [c for c in df.columns if any(k in c.lower() for k in ("label", "target", "y_true", "actual"))]

        left, right = st.columns([1, 2])

        with left:
            st.markdown("#### Filters & Controls")
            st.write(f"Rows in file: **{len(df):,}**")

            total_rows = len(df)
            positive_rate = None
            if prediction_candidates:
                pred_col = prediction_candidates[0]
                try:
                    numeric_pred = pd.to_numeric(df[pred_col], errors='coerce')
                    if numeric_pred.dropna().between(0, 1).all():
                        positive_rate = numeric_pred.mean()
                except Exception:
                    positive_rate = None

            c1, c2 = st.columns(2)
            c1.metric("Total rows", f"{total_rows:,}")
            if positive_rate is not None:
                c2.metric("Mean predicted positive", f"{positive_rate:.3f}")
            else:
                c2.metric("Prediction column", prediction_candidates[0] if prediction_candidates else "Not found")

            cols = list(df.columns)
            default_cols = cols[:min(10, len(cols))]
            display_cols = st.multiselect("Columns to display", cols, default=default_cols)

            sample_n = st.number_input("Max rows to preview", min_value=10, max_value=min(100000, len(df)), value=200, step=10)

            # REMOVED SEARCH BOX COMPLETELY

            column = st.selectbox("Column-specific filter", ["-- None --"] + cols)
            df_filtered = df.copy()

            if column and column != "-- None --":
                if pd.api.types.is_numeric_dtype(df[column]):
                    min_v = float(df[column].min())
                    max_v = float(df[column].max())
                    rng = st.slider("Value range", min_v, max_v, (min_v, max_v))
                    df_filtered = df_filtered[df_filtered[column].between(rng[0], rng[1])]
                else:
                    vals = sorted(df[column].astype(str).unique())
                    sel_vals = st.multiselect("Select values", vals, default=vals[:min(6, len(vals))])
                    if sel_vals:
                        df_filtered = df_filtered[df_filtered[column].astype(str).isin(sel_vals)]

            with st.expander("Advanced filters"):
                st.write("Add any quick transformations or preview sorting here.")
                sort_col = st.selectbox("Sort by (optional)", ["-- None --"] + cols)
                if sort_col and sort_col != "-- None --":
                    ascending = st.checkbox("Ascending order", value=False)
                    try:
                        df_filtered = df_filtered.sort_values(sort_col, ascending=ascending)
                    except Exception:
                        st.warning("Could not sort by that column (mixed types).")

            st.markdown("---")
            st.metric("Rows (filtered)", f"{len(df_filtered):,}")
            csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button("Download filtered CSV", csv_bytes, file_name="filtered_predictions.csv")

        with right:
            st.markdown("#### Data preview & charts")

            tab_table, tab_charts, tab_summary = st.tabs(["Table", "Charts", "Summary"])

            with tab_table:
                st.markdown("**Preview (interactive)**")
                if display_cols:
                    display_df = df_filtered[display_cols].head(sample_n)
                else:
                    display_df = df_filtered.head(sample_n)
                st.dataframe(display_df, height=480)

            with tab_charts:
                st.markdown("**Quick numeric summaries**")
                numeric_cols = df_filtered.select_dtypes(include=["number"]).columns.tolist()
                if numeric_cols:
                    ch1, ch2 = st.columns(2)
                    with ch1:
                        col_for_hist = st.selectbox("Histogram column", ["-- None --"] + numeric_cols, index=0)
                        if col_for_hist != "-- None --":
                            st.markdown(f"Histogram for **{col_for_hist}**")
                            st.bar_chart(df_filtered[col_for_hist].dropna().value_counts().sort_index().head(100))
                    with ch2:
                        col_for_box = st.selectbox("Top values column", ["-- None --"] + numeric_cols, index=0)
                        if col_for_box != "-- None --":
                            st.markdown(f"Top values for **{col_for_box}**")
                            st.write(df_filtered[col_for_box].value_counts().head(10))
                else:
                    st.info("No numeric columns available for charts.")

            with tab_summary:
                st.markdown("**Filtered summary**")
                st.write(f"Filtered rows: **{len(df_filtered):,}**")
                if prediction_candidates:
                    pred_col = prediction_candidates[0]
                    numeric_pred = pd.to_numeric(df_filtered[pred_col], errors='coerce')
                    if not numeric_pred.dropna().empty and numeric_pred.dropna().between(0, 1).all():
                        st.write(f"Mean predicted positive (filtered): **{numeric_pred.mean():.3f}**")
                        st.write(f"Median predicted positive (filtered): **{numeric_pred.median():.3f}**")
                    else:
                        st.write(f"Using prediction column: **{pred_col}** (not probability-like)")
                if label_candidates:
                    lbl = label_candidates[0]
                    st.write(f"Detected label/target column: **{lbl}**")

# -----------------------------------------------------
# Dynamic folder pages (unchanged behavior)
# -----------------------------------------------------

# ---- Deep Learning with Tensorflow ----
elif page == DL_FOLDER:
    st.title(DL_FOLDER)
    st.write(
        "The notebook and Python script implement a complete deep-learning pipeline for binary customer "
        "classification using TensorFlow and Keras. The materials include data preprocessing (scaling and "
        "label encoding), strategies for handling class imbalance such as SMOTE, model-building with a few "
        "different bias initializations, training and validation monitoring, and visual diagnostics (loss, AUC, "
        "precision/recall, confusion matrices). The workflow culminates in producing final prediction files for "
        "downstream use and deployment."
    )

    st.markdown("---")
    st.subheader("Jupyter Notebook")
    if dl_notebooks:
        for nb_path in dl_notebooks:
            nb_name = os.path.basename(nb_path)
            st.write(f"**{nb_name}**")
            try:
                with open(nb_path, "rb") as f:
                    st.download_button(label=f"Download {nb_name}", data=f.read(), file_name=nb_name, mime="application/octet-stream")
            except Exception as e:
                st.error(f"Could not prepare download for {nb_name}: {e}")
    else:
        st.info("No Jupyter notebook found for this folder.")

    if dl_scripts:
        st.markdown("---")
        st.subheader("Python Script")
        for sc_path in dl_scripts:
            sc_name = os.path.basename(sc_path)
            st.write(f"**{sc_name}**")
            try:
                with open(sc_path, "r", encoding="utf-8", errors="ignore") as f:
                    preview = "".join([next(f) for _ in range(200)])
                with st.expander("Preview script (first ~200 lines)"):
                    st.code(preview, language="python")
            except Exception:
                pass
            try:
                with open(sc_path, "rb") as f:
                    st.download_button(label=f"Download {sc_name}", data=f.read(), file_name=sc_name, mime="text/plain")
            except Exception as e:
                st.error(f"Could not prepare download for {sc_name}: {e}")

# ---- ETL process ----
elif page == ETL_FOLDER:
    st.title(ETL_FOLDER)
    st.write(
        "The ETL process notebook performs extraction, cleaning, transformation and loading of raw transaction "
        "and user data to produce a modeling-ready dataset. Steps include handling missing values and infinities, "
        "standardizing formats, encoding categorical fields, and basic profiling/validation. The notebook ensures "
        "data integrity and outputs structured files used for subsequent feature engineering and model training."
    )

    st.markdown("---")
    st.subheader("Jupyter Notebook")
    if etl_notebooks:
        for nb_path in etl_notebooks:
            nb_name = os.path.basename(nb_path)
            st.write(f"**{nb_name}**")
            try:
                with open(nb_path, "rb") as f:
                    st.download_button(label=f"Download {nb_name}", data=f.read(), file_name=nb_name, mime="application/octet-stream")
            except Exception as e:
                st.error(f"Could not prepare download for {nb_name}: {e}")
    else:
        st.info("No Jupyter notebook found for this folder.")

    if etl_scripts:
        st.markdown("---")
        st.subheader("Python Script")
        for sc_path in etl_scripts:
            sc_name = os.path.basename(sc_path)
            st.write(f"**{sc_name}**")
            try:
                with open(sc_path, "r", encoding="utf-8", errors="ignore") as f:
                    preview = "".join([next(f) for _ in range(200)])
                with st.expander("Preview script (first ~200 lines)"):
                    st.code(preview, language="python")
            except Exception:
                pass
            try:
                with open(sc_path, "rb") as f:
                    st.download_button(label=f"Download {sc_name}", data=f.read(), file_name=sc_name, mime="text/plain")
            except Exception as e:
                st.error(f"Could not prepare download for {sc_name}: {e}")

# ---- Exploratory Data Analysis ----
elif page == EDA_FOLDER:
    st.title(EDA_FOLDER)
    st.write(
        "This exploratory analysis notebook performs initial data checks and visual summaries to understand "
        "customer behavior and outcome distributions. It includes descriptive statistics, missing-value analysis, "
        "basic visualizations (histograms, jointplots), and early feature recoding ideas. The notebook helps "
        "identify anomalies, relevant predictors, and provides guidance for feature engineering and modeling."
    )

    st.markdown("---")
    st.subheader("Jupyter Notebook")
    if eda_notebooks:
        for nb_path in eda_notebooks:
            nb_name = os.path.basename(nb_path)
            st.write(f"**{nb_name}**")
            try:
                with open(nb_path, "rb") as f:
                    st.download_button(label=f"Download {nb_name}", data=f.read(), file_name=nb_name, mime="application/octet-stream")
            except Exception as e:
                st.error(f"Could not prepare download for {nb_name}: {e}")
    else:
        st.info("No Jupyter notebook found for this folder.")

    if eda_scripts:
        st.markdown("---")
        st.subheader("Python Script")
        for sc_path in eda_scripts:
            sc_name = os.path.basename(sc_path)
            st.write(f"**{sc_name}**")
            try:
                with open(sc_path, "r", encoding="utf-8", errors="ignore") as f:
                    preview = "".join([next(f) for _ in range(200)])
                with st.expander("Preview script (first ~200 lines)"):
                    st.code(preview, language="python")
            except Exception:
                pass
            try:
                with open(sc_path, "rb") as f:
                    st.download_button(label=f"Download {sc_name}", data=f.read(), file_name=sc_name, mime="text/plain")
            except Exception as e:
                st.error(f"Could not prepare download for {sc_name}: {e}")

# ---- Feature_generation ----
elif page == FEATURE_FOLDER:
    st.title(FEATURE_FOLDER)
    st.write(
        "The feature generation notebooks create new predictor variables derived from transactional and user-level "
        "data. Typical steps include aggregations (counts, rates), recoding categorical variables, time-windowed "
        "features, and scaling/normalization where appropriate. These notebooks serve as the repository of feature "
        "engineering experiments that feed into model training and cross-validation."
    )

    st.markdown("---")
    st.subheader("Jupyter Notebooks")
    if feature_notebooks:
        for nb_path in feature_notebooks:
            nb_name = os.path.basename(nb_path)
            st.write(f"**{nb_name}**")
            try:
                with open(nb_path, "rb") as f:
                    st.download_button(label=f"Download {nb_name}", data=f.read(), file_name=nb_name, mime="application/octet-stream")
            except Exception as e:
                st.error(f"Could not prepare download for {nb_name}: {e}")
    else:
        st.info("No feature generation notebooks found for this folder.")

    if feature_scripts:
        st.markdown("---")
        st.subheader("Python Script")
        for sc_path in feature_scripts:
            sc_name = os.path.basename(sc_path)
            st.write(f"**{sc_name}**")
            try:
                with open(sc_path, "r", encoding="utf-8", errors="ignore") as f:
                    preview = "".join([next(f) for _ in range(200)])
                with st.expander("Preview script (first ~200 lines)"):
                    st.code(preview, language="python")
            except Exception:
                pass
            try:
                with open(sc_path, "rb") as f:
                    st.download_button(label=f"Download {sc_name}", data=f.read(), file_name=sc_name, mime="text/plain")
            except Exception as e:
                st.error(f"Could not prepare download for {sc_name}: {e}")

# ---- Model fit & CV ----
elif page == MODEL_FOLDER:
    st.title(MODEL_FOLDER)
    st.write(
        "The model-fitting notebooks include training pipelines, cross-validation routines, and performance "
        "evaluation for candidate models. They typically perform label encoding, scaling, model construction "
        "(neural nets, or other classifiers), training loops, and k-fold or sampler-based CV to assess generalization. "
        "Plots and metric summaries help compare approaches and select a final model for production."
    )

    st.markdown("---")
    st.subheader("Jupyter Notebook")
    if model_notebooks:
        for nb_path in model_notebooks:
            nb_name = os.path.basename(nb_path)
            st.write(f"**{nb_name}**")
            try:
                with open(nb_path, "rb") as f:
                    st.download_button(label=f"Download {nb_name}", data=f.read(), file_name=nb_name, mime="application/octet-stream")
            except Exception as e:
                st.error(f"Could not prepare download for {nb_name}: {e}")
    else:
        st.info("No model-fitting notebook found for this folder.")

    if model_scripts:
        st.markdown("---")
        st.subheader("Python Script")
        for sc_path in model_scripts:
            sc_name = os.path.basename(sc_path)
            st.write(f"**{sc_name}**")
            try:
                with open(sc_path, "r", encoding="utf-8", errors="ignore") as f:
                    preview = "".join([next(f) for _ in range(200)])
                with st.expander("Preview script (first ~200 lines)"):
                    st.code(preview, language="python")
            except Exception:
                pass
            try:
                with open(sc_path, "rb") as f:
                    st.download_button(label=f"Download {sc_name}", data=f.read(), file_name=sc_name, mime="text/plain")
            except Exception as e:
                st.error(f"Could not prepare download for {sc_name}: {e}")

# -----------------------------------------------------
# Images pages (last in the sidebar order)
# -----------------------------------------------------
elif page == "Exploratory Images":
    render_image_selector_page("Exploratory Data — Images", images_for_exploratory)

elif page == "Feature Engineering":
    render_image_selector_page("Feature Engineering Visuals", images_for_feature_engineering)

elif page == "Model Results":
    render_image_selector_page("Model Results & Feature Importance", images_for_model_results)

# -----------------------------------------------------
# End of file
# -----------------------------------------------------
