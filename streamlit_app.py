import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, ConfusionMatrixDisplay
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #2a2a2a;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #888 !important; font-size: 12px !important; letter-spacing: 1px; text-transform: uppercase; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #f59e0b !important; font-family: 'IBM Plex Mono', monospace; font-size: 2rem !important; }

    /* Section headers */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #f59e0b;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    h1 { font-family: 'IBM Plex Mono', monospace !important; color: #f0f0f0 !important; }
    h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; color: #f0f0f0 !important; }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 1px;
        color: #888 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] { color: #f59e0b !important; border-bottom-color: #f59e0b !important; }

    /* Dataframe */
    .dataframe { background: #1a1a1a !important; }

    /* Buttons */
    .stButton > button {
        background: #f59e0b;
        color: #0f0f0f;
        border: none;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 1px;
        padding: 10px 24px;
    }
    .stButton > button:hover { background: #d97706; }

    /* Info box */
    .info-box {
        background: #1a1a1a;
        border-left: 3px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 6px 6px 0;
        margin: 12px 0;
        font-size: 14px;
        color: #ccc;
    }
    .tag {
        display: inline-block;
        background: #2a2a2a;
        color: #f59e0b;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 3px;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper: set dark plot style ───────────────────────────────────────────────
def dark_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#888', labelsize=9)
    ax.xaxis.label.set_color('#888')
    ax.yaxis.label.set_color('#888')
    ax.title.set_color('#f0f0f0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a2a')
    return fig, ax

# ── Load & merge data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_files):
    dfs = {f.name.replace('.csv','').lower(): pd.read_csv(f) for f in uploaded_files}
    # auto-detect which file is which
    general = survey = manager = None
    for name, d in dfs.items():
        if 'attrition' in d.columns or 'age' in d.columns.str.lower().tolist():
            general = d
        elif 'jobsatisfaction' in d.columns.str.lower().str.replace(' ','').tolist() or 'environmentsatisfaction' in d.columns.str.lower().str.replace(' ','').tolist():
            survey = d
        elif 'jobinvolvement' in d.columns.str.lower().str.replace(' ','').tolist() or 'performancerating' in d.columns.str.lower().str.replace(' ','').tolist():
            manager = d
    if general is None:
        # fallback: just concat / use whatever was uploaded
        combined = list(dfs.values())
        df = combined[0]
        for extra in combined[1:]:
            common = list(set(df.columns) & set(extra.columns))
            if common:
                df = df.merge(extra, on=common[0], suffixes=('', '_dup'))
        return df
    df = general
    if survey is not None:
        common = list(set(df.columns) & set(survey.columns))
        df = df.merge(survey, on=common[0] if common else df.columns[0])
    if manager is not None:
        common = list(set(df.columns) & set(manager.columns))
        df = df.merge(manager, on=common[0] if common else df.columns[0])
    return df

@st.cache_data
def preprocess(df):
    drop_cols = ['EmployeeID', 'EmployeeCount', 'StandardHours', 'Over18']
    df_c = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df_c['Attrition'] = df_c['Attrition'].map({'Yes': 1, 'No': 0})
    num_cols = df_c.select_dtypes(include=[np.number]).columns
    df_c[num_cols] = df_c[num_cols].fillna(df_c[num_cols].median())
    le = LabelEncoder()
    for col in df_c.select_dtypes(include=['object']).columns:
        df_c[col] = le.fit_transform(df_c[col].astype(str))
    return df_c

@st.cache_data
def train_models(df_clean):
    X = df_clean.drop(columns=['Attrition'])
    y = df_clean['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_tr_sc, y_train)
    y_lr = lr.predict(X_te_sc)

    dt = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_test if len(y_test)==len(X_train) else y_train)
    dt.fit(X_train, y_train)
    y_dt = dt.predict(X_test)

    return lr, dt, scaler, X, y, X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc, y_lr, y_dt

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 👥 Attrition Predictor")
    st.markdown("<div class='section-header'>CIS 412 · Code Blooded</div>", unsafe_allow_html=True)
    st.caption("Ella Myers · Nikasha Kapadia · Le Thu Trang Truong")
    st.divider()

    st.markdown("<div class='section-header'>Upload Dataset</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>Upload the 3 CSV files from the Kaggle dataset:<br>
    <span class='tag'>general_data.csv</span>
    <span class='tag'>employee_survey_data.csv</span>
    <span class='tag'>manager_survey_data.csv</span>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose CSV files", type="csv", accept_multiple_files=True, label_visibility="collapsed")

    if uploaded:
        st.success(f"✓ {len(uploaded)} file(s) loaded")

    st.divider()
    st.markdown("<div class='section-header'>Model Settings</div>", unsafe_allow_html=True)
    dt_depth   = st.slider("Decision Tree Max Depth", 2, 10, 5)
    test_size  = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    show_tree  = st.checkbox("Show Tree Visualization", value=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# Employee Attrition Predictor")
st.markdown("<div class='section-header'>CIS 412 Team Project · Phase 1 · CRISP-DM Framework</div>", unsafe_allow_html=True)

if not uploaded:
    st.markdown("""
    <div class='info-box'>
    ⬅️ Upload your CSV files in the sidebar to get started.<br><br>
    Download the dataset from Kaggle:
    <a href='https://www.kaggle.com/datasets/vjchoudhary7/hr-analytics-case-study' target='_blank' style='color:#f59e0b'>
    vjchoudhary7 / hr-analytics-case-study</a>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📋 Problem")
        st.write("Predict which employees are likely to leave using supervised classification.")
    with col2:
        st.markdown("### 🤖 Models")
        st.write("Logistic Regression and Decision Tree, evaluated on accuracy, precision, and recall.")
    with col3:
        st.markdown("### 💼 Value")
        st.write("Enables proactive HR retention strategies and data-driven workforce planning.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw   = load_data(uploaded)
df_clean = preprocess(df_raw)

# retrain with sidebar settings
X = df_clean.drop(columns=['Attrition'])
y = df_clean['Attrition']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y)
scaler    = StandardScaler()
X_tr_sc   = scaler.fit_transform(X_train)
X_te_sc   = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_tr_sc, y_train)
y_lr = lr.predict(X_te_sc)

dt = DecisionTreeClassifier(max_depth=dt_depth, random_state=42, class_weight='balanced')
dt.fit(X_train, y_train)
y_dt = dt.predict(X_test)

# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📊 DATA OVERVIEW", "🔍 EDA", "⚙️ PREPROCESSING", "🤖 MODELS", "📈 EVALUATION", "🔮 PREDICT"])

# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)

    attrition_col = df_raw['Attrition']
    if pd.api.types.is_numeric_dtype(attrition_col):
        attrition_rate = attrition_col.mean()
    else:
        attrition_rate = attrition_col.map({'Yes': 1, 'No': 0}).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", f"{len(df_raw):,}")
    c2.metric("Features", f"{df_raw.shape[1]}")
    c3.metric("Attrition Rate", f"{attrition_rate*100:.1f}%")
    c4.metric("Missing Values", f"{df_raw.isnull().sum().sum():,}")

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Raw Data Preview**")
        st.dataframe(df_raw.head(20), use_container_width=True, height=350)
    with col2:
        st.markdown("**Missing Values by Column**")
        mv = df_raw.isnull().sum()
        mv = mv[mv > 0].reset_index()
        mv.columns = ['Column', 'Missing']
        if len(mv):
            st.dataframe(mv, use_container_width=True)
        else:
            st.success("No missing values found!")
        st.markdown("**Data Types**")
        dt_counts = df_raw.dtypes.value_counts().reset_index()
        dt_counts.columns = ['Type', 'Count']
        st.dataframe(dt_counts, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    # Attrition distribution
    col1, col2 = st.columns(2)
    with col1:
        counts = df_raw['Attrition'].value_counts()
        fig, ax = dark_fig(5, 4)
        bars = ax.bar(counts.index, counts.values,
                      color=['#f59e0b', '#374151'], edgecolor='#2a2a2a', width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+15, str(val),
                    ha='center', color='#f0f0f0', fontweight='bold', fontsize=11)
        ax.set_title('Attrition Distribution', fontsize=13)
        ax.set_ylabel('Count', color='#888')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        # Pie
        fig, ax = dark_fig(5, 4)
        ax.pie(counts.values, labels=counts.index,
               colors=['#f59e0b', '#374151'],
               autopct='%1.1f%%', startangle=90,
               textprops={'color': '#f0f0f0'})
        ax.set_title('Attrition Share', fontsize=13)
        fig.patch.set_facecolor('#1a1a1a')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.divider()

    # Categorical features
    st.markdown("**Attrition by Categorical Features**")
    cat_options = df_raw.select_dtypes(include='object').columns.tolist()
    cat_options = [c for c in cat_options if c != 'Attrition']
    selected_cats = st.multiselect("Select features", cat_options,
                                   default=cat_options[:4])
    if selected_cats:
        cols = st.columns(min(2, len(selected_cats)))
        for i, col in enumerate(selected_cats):
            with cols[i % 2]:
                ct = df_raw.groupby([col, 'Attrition']).size().unstack(fill_value=0)
                fig, ax = dark_fig(6, 4)
                ct.plot(kind='bar', ax=ax, color=['#f59e0b', '#374151'],
                        edgecolor='#0f0f0f', rot=30, width=0.7)
                ax.set_title(f'Attrition by {col}', fontsize=11)
                ax.legend(title='Attrition', labelcolor='#888',
                          facecolor='#1a1a1a', edgecolor='#2a2a2a')
                ax.set_xlabel('')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    st.divider()

    # Numeric distributions
    st.markdown("**Numeric Feature Distributions by Attrition**")
    num_options = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    sel_nums = st.multiselect("Select numeric features", num_options,
                              default=['Age','MonthlyIncome','YearsAtCompany'] if all(c in num_options for c in ['Age','MonthlyIncome','YearsAtCompany']) else num_options[:3])
    if sel_nums:
        cols = st.columns(min(3, len(sel_nums)))
        for i, col in enumerate(sel_nums):
            with cols[i % 3]:
                fig, ax = dark_fig(4, 3)
                for label, color in zip(['Yes','No'], ['#f59e0b','#374151']):
                    subset = df_raw[df_raw['Attrition']==label][col].dropna()
                    ax.hist(subset, bins=20, alpha=0.7, color=color,
                            label=f'Attrition={label}', edgecolor='#0f0f0f')
                ax.set_title(col, fontsize=10)
                ax.legend(fontsize=7, facecolor='#1a1a1a',
                          edgecolor='#2a2a2a', labelcolor='#888')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    st.divider()

    # Correlation heatmap
    st.markdown("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    num_df = df_raw.select_dtypes(include=[np.number])
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), ax=ax, annot=False, cmap='YlOrBr',
                center=0, mask=mask, linewidths=0.3,
                cbar_kws={'shrink': 0.8})
    ax.tick_params(colors='#888', labelsize=8)
    ax.set_title('Feature Correlation Matrix', color='#f0f0f0', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-header'>Data Preparation</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Steps Applied**")
        steps = [
            ("🗑️ Dropped columns", "EmployeeID, EmployeeCount, StandardHours, Over18"),
            ("🎯 Target encoded", "Attrition: Yes→1, No→0"),
            ("🔢 Missing values", "Imputed with column median"),
            ("🏷️ Categorical encoding", "LabelEncoder on all object columns"),
            ("✂️ Train/Test split", f"{int((1-test_size)*100)}/{int(test_size*100)} stratified"),
            ("📐 Feature scaling", "StandardScaler (Logistic Regression only)"),
        ]
        for icon_label, detail in steps:
            st.markdown(f"**{icon_label}:** {detail}")

    with col2:
        st.markdown("**Class Balance**")
        cb = y.value_counts().reset_index()
        cb.columns = ['Attrition', 'Count']
        cb['Label'] = cb['Attrition'].map({0:'No Attrition', 1:'Attrition'})
        cb['Pct'] = (cb['Count'] / cb['Count'].sum() * 100).round(1).astype(str) + '%'
        st.dataframe(cb[['Label','Count','Pct']], use_container_width=True, hide_index=True)

        st.markdown("**Split Summary**")
        split_df = pd.DataFrame({
            'Set': ['Train', 'Test'],
            'Size': [len(X_train), len(X_test)],
            'Pct': [f'{int((1-test_size)*100)}%', f'{int(test_size*100)}%']
        })
        st.dataframe(split_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Processed Feature Preview**")
    st.dataframe(df_clean.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("<div class='section-header'>Model Results</div>", unsafe_allow_html=True)

    model_tab1, model_tab2 = st.tabs(["📉 LOGISTIC REGRESSION", "🌳 DECISION TREE"])

    with model_tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Confusion Matrix**")
            fig, ax = dark_fig(5, 4)
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_lr,
                display_labels=['No Attrition', 'Attrition'],
                cmap='YlOrBr', ax=ax)
            ax.set_title('Logistic Regression', color='#f0f0f0')
            fig.patch.set_facecolor('#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("**Classification Report**")
            report = classification_report(y_test, y_lr,
                        target_names=['No Attrition','Attrition'], output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        st.markdown("**Top 15 Feature Coefficients**")
        coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr.coef_[0]})
        coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index).head(15)
        fig, ax = dark_fig(10, 5)
        colors = ['#f59e0b' if c > 0 else '#60a5fa' for c in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'],
                color=colors, edgecolor='#0f0f0f')
        ax.axvline(0, color='#555', linewidth=0.8)
        ax.set_title('Feature Coefficients (amber=increases risk, blue=decreases risk)', fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with model_tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Confusion Matrix**")
            fig, ax = dark_fig(5, 4)
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_dt,
                display_labels=['No Attrition', 'Attrition'],
                cmap='YlOrBr', ax=ax)
            ax.set_title('Decision Tree', color='#f0f0f0')
            fig.patch.set_facecolor('#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("**Classification Report**")
            report_dt = classification_report(y_test, y_dt,
                            target_names=['No Attrition','Attrition'], output_dict=True)
            st.dataframe(pd.DataFrame(report_dt).T.round(3), use_container_width=True)

        st.markdown("**Top 15 Feature Importances**")
        fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': dt.feature_importances_})
        fi_df = fi_df.sort_values('Importance', ascending=False).head(15)
        fig, ax = dark_fig(10, 5)
        ax.barh(fi_df['Feature'], fi_df['Importance'],
                color='#f59e0b', edgecolor='#0f0f0f')
        ax.set_title('Feature Importances', fontsize=11)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        if show_tree:
            st.markdown("**Decision Tree Visualization (top 3 levels)**")
            fig, ax = plt.subplots(figsize=(22, 8))
            fig.patch.set_facecolor('#1a1a1a')
            plot_tree(dt, max_depth=3, feature_names=X.columns,
                      class_names=['No Attrition','Attrition'],
                      filled=True, rounded=True, fontsize=8, ax=ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-header'>Model Evaluation & Comparison</div>", unsafe_allow_html=True)

    results = pd.DataFrame({
        'Model':     ['Logistic Regression', 'Decision Tree'],
        'Accuracy':  [accuracy_score(y_test,y_lr),  accuracy_score(y_test,y_dt)],
        'Precision': [precision_score(y_test,y_lr), precision_score(y_test,y_dt)],
        'Recall':    [recall_score(y_test,y_lr),    recall_score(y_test,y_dt)],
        'F1 Score':  [f1_score(y_test,y_lr),        f1_score(y_test,y_dt)],
    }).set_index('Model').round(4)

    # Highlight best per column
    st.dataframe(results.style.highlight_max(axis=0, color='#78350f'), use_container_width=True)

    st.divider()

    # Bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = dark_fig(10, 5)
    b1 = ax.bar(x-w/2, results.loc['Logistic Regression', metrics],
                w, label='Logistic Regression', color='#f59e0b', edgecolor='#0f0f0f')
    b2 = ax.bar(x+w/2, results.loc['Decision Tree', metrics],
                w, label='Decision Tree', color='#60a5fa', edgecolor='#0f0f0f')
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_title('Model Comparison', fontsize=13)
    ax.legend(facecolor='#1a1a1a', edgecolor='#2a2a2a', labelcolor='#f0f0f0')
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=9, color='#f0f0f0')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()
    st.markdown("""<div class='info-box'>
    <strong>Key Insight:</strong> For HR attrition, <strong>Recall</strong> is the most important metric.
    Missing an employee who is about to leave (false negative) is more costly than a false positive.
    Choose the model with the highest recall for production use.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("<div class='section-header'>Predict Individual Employee Attrition Risk</div>", unsafe_allow_html=True)
    st.markdown("Enter employee details to get a real-time attrition risk score.")

    model_choice = st.radio("Model", ["Logistic Regression", "Decision Tree"], horizontal=True)

    # Build input form from top features
    top_features = X.columns.tolist()[:12]

    col1, col2, col3 = st.columns(3)
    input_vals = {}
    for i, feat in enumerate(top_features):
        col = [col1, col2, col3][i % 3]
        mn = float(X[feat].min()); mx = float(X[feat].max())
        med = float(X[feat].median())
        with col:
            input_vals[feat] = st.number_input(feat, min_value=mn, max_value=mx, value=med, key=feat)

    # Fill remaining features with median
    full_input = {col: float(X[col].median()) for col in X.columns}
    full_input.update(input_vals)
    input_df = pd.DataFrame([full_input])[X.columns]

    if st.button("⚡ Predict Attrition Risk"):
        if model_choice == "Logistic Regression":
            inp_sc = scaler.transform(input_df)
            prob = lr.predict_proba(inp_sc)[0][1]
            pred = lr.predict(inp_sc)[0]
        else:
            prob = dt.predict_proba(input_df)[0][1]
            pred = dt.predict(input_df)[0]

        st.divider()
        risk_label = "🔴 HIGH RISK" if pred == 1 else "🟢 LOW RISK"
        c1, c2 = st.columns(2)
        c1.metric("Prediction", risk_label)
        c2.metric("Attrition Probability", f"{prob*100:.1f}%")

        # Risk gauge
        fig, ax = dark_fig(6, 2)
        ax.barh(['Risk'], [prob], color='#f59e0b' if prob > 0.5 else '#22c55e',
                height=0.4, edgecolor='#0f0f0f')
        ax.barh(['Risk'], [1-prob], left=[prob], color='#2a2a2a',
                height=0.4, edgecolor='#0f0f0f')
        ax.set_xlim(0, 1)
        ax.set_title(f'Attrition Probability: {prob*100:.1f}%', fontsize=12)
        ax.axvline(0.5, color='#555', linestyle='--', linewidth=1)
        ax.text(0.51, 0, '50%', color='#888', fontsize=8, va='center')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        if pred == 1:
            st.markdown("""<div class='info-box'>
            ⚠️ <strong>Recommended Actions:</strong> Consider compensation review, career development discussion,
            workload assessment, or work-life balance initiatives for this employee.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='info-box'>
            ✅ <strong>Low attrition risk detected.</strong> Continue regular check-ins and maintain
            current engagement levels.
            </div>""", unsafe_allow_html=True)