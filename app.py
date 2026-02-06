"""
GradSingapore Survey Analytics Dashboard
======================================
A professional Streamlit dashboard for analyzing student survey data.
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
import os
import uuid
import time
import numpy as np
from sklearn.cluster import KMeans


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="GradSingapore Analytics",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CSS STYLES - Professional SaaS Theme
# =============================================================================

SAAS_CSS = """
<style>
    /* --- Color Palette --- */
    :root {
        --primary: #2563eb;
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-300: #d1d5db;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --gray-900: #111827;
    }

    /* --- Main Header --- */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 24px;
        font-weight: 700;
    }
    .main-header p {
        margin: 4px 0 0 0;
        opacity: 0.9;
        font-size: 14px;
    }

    /* --- Header Actions --- */
    .header-actions {
        display: flex;
        gap: 12px;
    }
    .header-btn {
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .header-btn:hover {
        background: rgba(255,255,255,0.3);
    }

    /* --- Chart Card --- */
    .chart-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        height: 100%;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .chart-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .chart-card h4 {
        margin: 0 0 16px 0;
        font-size: 15px;
        font-weight: 600;
        color: #1f2937;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* --- Insight Text --- */
    .insight-text {
        font-size: 12px;
        color: #6b7280;
        font-style: italic;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #f3f4f6;
        line-height: 1.5;
    }

    /* --- Placeholder Card --- */
    .placeholder-card {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 48px 24px;
        text-align: center;
        color: #6b7280;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .placeholder-card:hover {
        border-color: #2563eb;
        background: linear-gradient(135deg, #eff6ff 0%, #f3f4f6 100%);
    }
    .placeholder-card h4 {
        margin: 0;
        font-size: 14px;
        color: #9ca3af;
    }
    .placeholder-card p {
        margin: 8px 0 0 0;
        font-size: 12px;
    }

    /* --- Section Title --- */
    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #1f2937;
        margin: 0 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* --- Metric Cards --- */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1f2937;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f3f4f6;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 14px;
        color: #6b7280;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        color: #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    .stSidebar .stMarkdown h3 {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6b7280;
        margin-bottom: 12px;
    }

    /* --- Floating Chat Button --- */
    div.element-container:has(button[kind="primary"]) {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 1000;
    }
    div.element-container button[kind="primary"] {
        border-radius: 50%;
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        font-size: 20px;
        border: none;
        transition: all 0.2s;
    }
    div.element-container button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.5);
        transform: scale(1.05);
    }

    /* --- Filter Badge --- */
    .filter-badge {
        background: #eff6ff;
        color: #2563eb;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }
</style>
"""

st.markdown(SAAS_CSS, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I can help you analyze the survey data. Ask me about completion rates, specific schools, or attractiveness scores."
        }
    ]

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "insights" not in st.session_state:
    st.session_state.insights = {}


# =============================================================================
# DATA LOADING
# =============================================================================

COLUMN_RENAME_MAP = {
    'Response ID': 'id',
    'Time Started': 'start_time',
    'Date Submitted': 'submit_time',
    'Status': 'status',
    'User Agent': 'user_agent',
    'Country': 'country',
    'Which higher education institution do you or did you study at?': 'school',
    'What is your current year of study as of 2025?': 'year',
    'What will be your highest qualification when you graduate?': 'qualification',
    'Which of the following best describes the main subject that you are studying?\xa0': 'subject',
    'Please indicate your nationality.': 'nationality',
    'What is your gender?': 'gender',
    'Which of these statements best describes your current perception of the organisation as an employer?': 'perception',
    'Types of roles available:What do you wish to learn more about regarding the organisation as an employer? (Pick 3) \xa0': 'info_roles',
    'Career progression and development:What do you wish to learn more about regarding the organisation as an employer? (Pick 3) \xa0': 'info_career',
    'Compensation and benefits:What do you wish to learn more about regarding the organisation as an employer? (Pick 3) \xa0': 'info_comp',
    'Work-life balance and culture:What do you wish to learn more about regarding the organisation as an employer? (Pick 3) \xa0': 'info_culture',
    'Application and interview process:What do you wish to learn more about regarding the organisation as an employer? (Pick 3) \xa0': 'info_process',
    'Other - Write In (Required):What do you wish to learn more about regarding the organisation as an employer? (Pick 3) \xa0.1': 'info_other_text',
    'On a scale from 1 to 10 (1 – Low, 10 – High), how would you rate the attractiveness of the organisation as an employer?\xa0 \xa0': 'attractiveness',
    'Which of these factors would most motivate you to apply for a position at the organisation? \xa0': 'motivation',
    'Other - Write In (Required):Which of these factors would most motivate you to apply for a position at the organisation? \xa0': 'motivation_other'
}


@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Load and preprocess survey data."""
    file_path = "Category B Dataset/sds_datathon_gradsingapore.xlsx"
    
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return None

    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Feature engineering
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['submit_time'] = pd.to_datetime(df['submit_time'])
    df['duration_sec'] = (df['submit_time'] - df['start_time']).dt.total_seconds()
    df['is_mobile'] = df['user_agent'].str.contains('Mobile|Android|iPhone', case=False, na=False).astype(int)

    return df


# =============================================================================
# FILTER MANAGEMENT
# =============================================================================

def on_filter_change():
    """Callback when filters change."""
    st.session_state.chat_open = False
    st.session_state.insights = {}


def reset_schools():
    """Reset school filter to all."""
    st.session_state.school_selector = all_schools
    on_filter_change()


def reset_years():
    """Reset year filter to all."""
    st.session_state.year_selector = all_years
    on_filter_change()


# =============================================================================
# AI ANALYST
# =============================================================================

SYSTEM_PROMPT = """
You are a senior data analyst working on a student survey analytics platform (GradSingapore).

You are given a pandas DataFrame named `df` that contains survey responses.

CORE METADATA:
- id: unique response ID
- start_time: survey start timestamp
- submit_time: survey submission timestamp
- status: completion status (Complete, Partial, Disqualified)
- duration_sec: time spent in seconds
- user_agent: browser/device string
- is_mobile: 1 if mobile, 0 otherwise
- country: respondent country

DEMOGRAPHICS:
- school: higher education institution
- year: current year of study
- qualification: expected highest qualification
- subject: main field of study
- nationality: respondent nationality
- gender: respondent gender

EMPLOYER PERCEPTION:
- perception: overall perception as employer
- attractiveness: attractiveness score (1-10)
- motivation: main motivating factor
- motivation_other: free-text motivation

INFORMATION INTERESTS:
- info_roles: interest in types of roles
- info_career: interest in career progression
- info_comp: interest in compensation & benefits
- info_culture: interest in work-life balance & culture
- info_process: interest in application process
- info_other_text: free-text "other" interest

TASK:
- Answer user questions by analyzing `df`
- Focus on survey quality, engagement, drop-offs, segmentation, and employer attractiveness
- Prefer rates, distributions, comparisons, and patterns over raw counts

OUTPUT RULES:
- Output ONLY valid Python code
- Do NOT include explanations outside code
- Do NOT use markdown
- If you generate a plot:
    - call plt.clf() before plotting
    - save the figure to 'temp_plot.png'
    - do NOT call plt.show()
- Store the final explanation in: final_answer = "<clear explanation>"
"""


def build_analyst_agent(df):
    """Build the LLM analyst agent."""
    llm = ChatOpenAI(
        temperature=0,
        model=st.secrets.get("MODEL_NAME", "glm-4.7"),
        openai_api_key=st.secrets.get("OPENAI_API_KEY", "your-api-key"),
        openai_api_base=st.secrets.get("OPENAI_API_BASE", "https://api.z.ai/api/paas/v4/"),
    )
    return llm, SYSTEM_PROMPT


def run_analyst_agent(user_question: str, df) -> tuple:
    """Run the analyst agent with a user question."""
    llm, system_prompt = build_analyst_agent(df)
    
    prompt = f"{system_prompt}\n\nUser question:\n{user_question}"
    
    try:
        response = llm.invoke(prompt)
        code = response.content

        # Clean up code block markers
        if code.startswith("```python"):
            code = code.replace("```python", "").replace("```", "")
        elif code.startswith("```"):
            code = code.replace("```", "")
        code = code.strip()

        local_vars = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns}
        exec(code, {}, local_vars)

        final_answer = local_vars.get("final_answer", "Analysis completed.")

        # Handle generated plot
        image_path = None
        if os.path.exists("temp_plot.png"):
            unique_filename = f"chart_{uuid.uuid4().hex}.png"
            os.rename("temp_plot.png", unique_filename)
            image_path = unique_filename

        return final_answer, code, image_path

    except Exception as e:
        return f"Error: {str(e)}", code if 'code' in locals() else "", None


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the dashboard header."""
    st.markdown("""
    <div class="main-header">
        <div>
            <h1>📊 GradSingapore Survey Analytics</h1>
            <p>Student perception survey analysis dashboard</p>
        </div>
        <div class="header-actions">
            <button class="header-btn" onclick="alert('Export feature coming soon!')">📥 Export</button>
            <button class="header-btn" onclick="location.reload()">🔄 Refresh</button>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_filter_bar(schools_count: int, years_count: int, responses_count: int):
    """Render the filter status bar."""
    st.markdown(f"""
    <div class="chart-card" style="padding: 16px 20px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
            <span style="font-size: 13px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px;">Filters</span>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="filter-badge">Schools: {schools_count}</span>
                <span class="filter-badge">Years: {years_count}</span>
                <span class="filter-badge" style="background: #f0fdf4; color: #16a34a;">Responses: {responses_count}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_filters():
    """Render sidebar filter controls."""
    st.sidebar.title("🔍 Filters")
    
    # Generate insights button
    gen_insights = st.sidebar.button("✨ Generate Insights")
    
    st.sidebar.divider()
    
    # School filter
    st.sidebar.subheader("Higher Education School")
    selected_schools = st.sidebar.multiselect(
        "Select schools:",
        options=all_schools,
        key='school_selector',
        on_change=on_filter_change
    )
    st.sidebar.caption(f"{len(selected_schools)} of {len(all_schools)} selected")
    st.sidebar.button("Select All Schools", on_click=reset_schools)
    
    st.sidebar.divider()
    
    # Year filter
    st.sidebar.subheader("Year of Study")
    selected_years = st.sidebar.multiselect(
        "Select years:",
        options=all_years,
        key='year_selector',
        on_change=on_filter_change
    )
    st.sidebar.caption(f"{len(selected_years)} of {len(all_years)} selected")
    st.sidebar.button("Select All Years", on_click=reset_years)
    
    return gen_insights, selected_schools, selected_years


def render_sidebar_metrics(total_count: int, filtered_count: int):
    """Render sidebar metrics."""
    st.sidebar.divider()
    st.sidebar.metric("Filtered Responses", f"{filtered_count}")
    st.sidebar.metric("Total Responses", f"{total_count}")


def render_metrics(filtered_df: pd.DataFrame):
    """Render KPI metrics row."""
    k1, k2, k3, k4 = st.columns(4)
    
    partial_rate = 0
    completion_rate = 0
    
    if 'status' in filtered_df.columns:
        partial_rate = (filtered_df['status'] == 'Partial').mean() * 100
        completion_rate = (filtered_df['status'] == 'Complete').mean() * 100
    
    avg_attract = 0
    if 'attractiveness' in filtered_df.columns:
        avg_attract = filtered_df['attractiveness'].mean()
    
    k1.metric("Total Responses", f"{len(filtered_df)}")
    k2.metric("Completion Rate", f"{completion_rate:.1f}%")
    k3.metric("Partial Rate", f"{partial_rate:.1f}%")
    k4.metric("Avg Attractiveness", f"{avg_attract:.1f}")


def render_chart_card(title: str, chart_function, insight_key: str = None):
    """Helper to render a chart card with optional insight."""
    st.markdown(f"<div class='chart-card'><h4>{title}</h4>", unsafe_allow_html=True)
    chart_function()
    if insight_key and insight_key in st.session_state.insights:
        st.markdown(f"<p class='insight-text'>{st.session_state.insights[insight_key]}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# TAB 1: SURVEY QUALITY GRAPHS
# =============================================================================

def chart_status_distribution(filtered_df: pd.DataFrame):
    """Status Distribution (Donut Chart) - Cell 2656"""
    if 'status' not in filtered_df.columns:
        return
    
    status_colors = {
        'Complete': '#2E7D32',
        'Partial': '#FF9800',
        'Disqualified': '#D32F2F'
    }
    
    status_counts = filtered_df['status'].value_counts().reindex(status_colors.keys())
    
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    wedges = ax.pie(
        status_counts,
        labels=status_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=[status_colors[s] for s in status_counts.index],
        wedgeprops=dict(width=0.5)
    )
    ax.set_title('Survey Status Distribution', fontsize=12, fontweight='bold')
    st.pyplot(fig)


def chart_dropoff_analysis(filtered_df: pd.DataFrame):
    """Drop-off Analysis (Bar Chart) - Cell 3432"""
    if 'status' not in filtered_df.columns:
        return
    
    # Use 'school' instead of 'university' as per COLUMN_RENAME_MAP
    survey_questions = [
        'school', 'year', 'qualification', 'subject', 'nationality',
        'gender', 'perception', 'info_roles', 'info_career', 'info_comp',
        'info_culture', 'info_process', 'info_other_text', 'attractiveness',
        'motivation', 'motivation_other'
    ]
    
    # Filter to only questions that exist in the dataframe
    survey_questions = [q for q in survey_questions if q in filtered_df.columns]
    
    partial_df = filtered_df[filtered_df['status'] == 'Partial'].copy()
    
    if len(partial_df) == 0:
        st.markdown("<p style='text-align: center; color: #666;'>No partial responses to analyze</p>", unsafe_allow_html=True)
        return
    
    def first_blank_question(row):
        for q in survey_questions:
            if pd.isna(row[q]) or str(row[q]).strip() == "":
                return q
        return None
    
    partial_df['first_dropoff_question'] = partial_df.apply(first_blank_question, axis=1)
    
    dropoff_summary = partial_df['first_dropoff_question'].value_counts().reset_index()
    dropoff_summary.columns = ['question', 'num_dropoffs']
    dropoff_summary['percentage'] = (dropoff_summary['num_dropoffs'] / dropoff_summary['num_dropoffs'].sum() * 100).round(2)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    sns.barplot(
        data=dropoff_summary,
        x='num_dropoffs',
        y='question',
        hue='question',
        palette='Reds_r',
        ax=ax,
        legend=False
    )
    ax.set_title('First Drop-Off Point for Partial Survey Responses', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Respondents', fontsize=11)
    ax.set_ylabel('Question Where Respondents First Stopped', fontsize=11)
    sns.despine()
    st.pyplot(fig)


def chart_question_correlation_heatmap(filtered_df: pd.DataFrame):
    """Question Correlation Heatmap - Cell 2798"""
    info_cols = ['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']
    
    # Check if columns exist
    missing_cols = [col for col in info_cols if col not in filtered_df.columns]
    if missing_cols:
        st.markdown(f"<p style='text-align: center; color: #666;'>Missing columns: {', '.join(missing_cols)}</p>", unsafe_allow_html=True)
        return
    
    info_binary = filtered_df[info_cols].notna().astype(int)
    info_corr = info_binary.corr()
    
    info_rename_map = {
        'info_roles': 'Types of Roles',
        'info_career': 'Career Progression',
        'info_comp': 'Compensation & Benefits',
        'info_culture': 'Company Culture',
        'info_process': 'Application Process'
    }
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    sns.heatmap(
        info_corr.rename(index=info_rename_map, columns=info_rename_map),
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        ax=ax
    )
    ax.set_title('Redundancy Check: Learn More Correlations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)


def chart_partial_prediction_model(filtered_df: pd.DataFrame):
    """Partial Prediction Model Performance - Cells 3044-3320"""
    # Check if model files exist
    import os
    model_path = 'partial_response_model.pkl'
    encoders_path = 'partial_response_label_encoders.pkl'
    features_path = 'partial_response_feature_list.pkl'
    
    if not (os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(features_path)):
        # Fallback to metrics display if model files not found
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h4 style='color: #2E7D32;'>Model Performance Metrics</h4>
            <div style='display: flex; justify-content: center; gap: 30px; margin-top: 15px;'>
                <div style='background: #f5f5f5; padding: 15px 25px; border-radius: 8px;'>
                    <div style='font-size: 24px; font-weight: bold; color: #2E7D32;'>86.4%</div>
                    <div style='font-size: 12px; color: #666;'>Accuracy</div>
                </div>
                <div style='background: #f5f5f5; padding: 15px 25px; border-radius: 8px;'>
                    <div style='font-size: 24px; font-weight: bold; color: #FF9800;'>4.7%</div>
                    <div style='font-size: 12px; color: #666;'>Precision</div>
                </div>
                <div style='background: #f5f5f5; padding: 15px 25px; border-radius: 8px;'>
                    <div style='font-size: 24px; font-weight: bold; color: #2196F3;'>21.1%</div>
                    <div style='font-size: 12px; color: #666;'>Recall</div>
                </div>
                <div style='background: #f5f5f5; padding: 15px 25px; border-radius: 8px;'>
                    <div style='font-size: 24px; font-weight: bold; color: #9C27B0;'>7.6%</div>
                    <div style='font-size: 12px; color: #666;'>F1-Score</div>
                </div>
                <div style='background: #f5f5f5; padding: 15px 25px; border-radius: 8px;'>
                    <div style='font-size: 24px; font-weight: bold; color: #00BCD4;'>57.7%</div>
                    <div style='font-size: 12px; color: #666;'>ROC-AUC</div>
                </div>
            </div>
            <p style='font-size: 11px; color: #999; margin-top: 15px;'>Random Forest Model trained on respondent demographics</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    try:
        import joblib
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Load model and preprocessing objects
        rf = joblib.load(model_path)
        label_encoders = joblib.load(encoders_path)
        analysis_vars = joblib.load(features_path)
        
        # Rename 'school' to 'university' to match model expectations
        df_work = filtered_df.copy()
        if 'school' in df_work.columns and 'university' not in df_work.columns:
            df_work = df_work.rename(columns={'school': 'university'})
        
        # Prepare data - use original feature names from model training
        analysis_vars_local = [v for v in analysis_vars if v in df_work.columns]
        
        # Ensure all model features are present
        missing_features = [v for v in analysis_vars if v not in df_work.columns]
        if missing_features:
            st.markdown(f"<p style='text-align: center; color: #666;'>Missing features: {', '.join(missing_features)}</p>", unsafe_allow_html=True)
            return
        
        df_model = df_work[analysis_vars + ['status']].dropna().copy()
        
        if len(df_model) < len(analysis_vars) + 1:
            st.markdown("<p style='text-align: center; color: #666;'>Insufficient data for model visualization</p>", unsafe_allow_html=True)
            return
        
        # Target: Partial vs Not Partial
        df_model['is_partial'] = (df_model['status'] == 'Partial').astype(int)
        
        # Encode features using stored encoders
        X = df_model[analysis_vars].copy()
        for col in analysis_vars:
            if col in label_encoders:
                le = label_encoders[col]
                X[col] = X[col].astype(str)
                # Handle unseen labels by mapping to most common
                X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        
        y = df_model['is_partial']
        
        if len(y.unique()) < 2:
            st.markdown("<p style='text-align: center; color: #666;'>Insufficient class variation for model visualization</p>", unsafe_allow_html=True)
            return
        
        # Split and predict
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        y_pred = rf.predict(X_test)
        
        # Create 2-subplot figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
        
        # Left: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Not Partial', 'Partial']
        )
        disp.plot(ax=axes[0], cmap='Oranges', values_format='d')
        axes[0].set_title('Confusion Matrix: Partial vs Not Partial', fontsize=12, fontweight='bold')
        
        # Right: Feature Importance
        importances = pd.Series(rf.feature_importances_, index=analysis_vars).sort_values(ascending=True)
        
        axes[1].barh(importances.index, importances.values, color='#FFB347', edgecolor='black')
        axes[1].set_title('Predictive Importance: Factors Driving Partial Responses', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Importance Score', fontsize=11)
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.markdown(f"<p style='text-align: center; color: #666;'>Model visualization unavailable</p>", unsafe_allow_html=True)


# =============================================================================
# TAB 2: SURVEY RESULTS GRAPHS
# =============================================================================

def chart_interest_personas(filtered_df: pd.DataFrame):
    """Interest-Based Personas (Clustering) - Cell 3590"""
    info_cols = ['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']
    
    # Check required columns
    missing_cols = [col for col in info_cols if col not in filtered_df.columns]
    if missing_cols or 'subject' not in filtered_df.columns:
        st.markdown(f"<p style='text-align: center; color: #666;'>Missing required columns</p>", unsafe_allow_html=True)
        return
    
    clustering_data = filtered_df[info_cols + ['subject', 'attractiveness']].copy()
    clustering_data[info_cols] = clustering_data[info_cols].notnull().astype(int)
    clustering_clean = clustering_data.dropna(subset=['subject', 'attractiveness']).copy()
    
    if len(clustering_clean) < 3:
        st.markdown(f"<p style='text-align: center; color: #666;'>Insufficient data for clustering</p>", unsafe_allow_html=True)
        return
    
    # Perform K-Means Clustering
    X_cluster = clustering_clean[info_cols].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clustering_clean.loc[:, 'persona'] = kmeans.fit_predict(X_cluster)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=info_cols)
    
    # Create labels based on cluster centers
    interest_names = {
        'info_roles': 'Roles',
        'info_career': 'Career',
        'info_comp': 'Compensation',
        'info_culture': 'Culture',
        'info_process': 'Process'
    }
    
    persona_labels = []
    for idx, row in cluster_centers.iterrows():
        top_two = row.nlargest(2)
        top_name = interest_names[top_two.index[0]]
        second_name = interest_names[top_two.index[1]]
        persona_labels.append(f"{top_name} & {second_name} Focused")
    
    persona_mapping = {i: persona_labels[i] for i in range(3)}
    clustering_clean.loc[:, 'persona_label'] = clustering_clean['persona'].map(persona_mapping)
    
    # Create 2-subplot figure matching notebook
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    # Left: Cluster centers heatmap
    sns.heatmap(
        cluster_centers.T,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=[persona_labels[i] for i in range(3)],
        yticklabels=['Roles', 'Career', 'Compensation', 'Culture', 'Process'],
        ax=axes[0],
        cbar_kws={'label': 'Interest Intensity'},
        annot_kws={'size': 10, 'weight': 'bold'}
    )
    axes[0].set_title('Interest-Based Personas: Cluster Profiles', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Persona Type', fontsize=11)
    axes[0].set_ylabel('Interest Categories', fontsize=11)
    
    # Right: Persona distribution
    persona_counts = clustering_clean['persona_label'].value_counts()
    colors_persona = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    axes[1].barh(persona_counts.index, persona_counts.values, color=colors_persona, edgecolor='black')
    axes[1].set_title('Distribution of Interest-Based Personas', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Respondents', fontsize=11)
    axes[1].set_ylabel('Persona Type', fontsize=11)
    
    for i, v in enumerate(persona_counts.values):
        axes[1].text(v + 5, i, str(v), va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)


def chart_motivation_drivers(filtered_df: pd.DataFrame):
    """Motivation Drivers (Box Plot) - Cell 4481"""
    if 'attractiveness' not in filtered_df.columns or 'motivation' not in filtered_df.columns:
        return
    
    driver_data = filtered_df[(filtered_df['status'] == 'Complete') & 
                             (filtered_df['attractiveness'].notna()) & 
                             (filtered_df['motivation'].notna())].copy()
    
    if len(driver_data) == 0:
        st.markdown("<p style='text-align: center; color: #666;'>No complete responses with motivation data</p>", unsafe_allow_html=True)
        return
    
    # Calculate average attractiveness by motivation
    motivation_attractiveness = driver_data.groupby('motivation')['attractiveness'].agg(['mean', 'count']).round(2)
    motivation_attractiveness = motivation_attractiveness.sort_values('mean', ascending=True)
    
    # Get top motivations by count
    top_motivations = motivation_attractiveness.nlargest(8, 'count').index
    motivation_top = motivation_attractiveness[motivation_attractiveness.index.isin(top_motivations)]
    
    # Create 2-subplot figure matching notebook
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    
    # Left: Bar chart with gradient colors (matching notebook: plt.cm.RdYlGn)
    colors_gradient = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(motivation_top)))
    bars = axes[0].barh(motivation_top.index, motivation_top['mean'], color=colors_gradient, edgecolor='black')
    axes[0].set_xlabel('Average Attractiveness Rating (1-10)', fontsize=11)
    axes[0].set_ylabel('Motivation Factor', fontsize=11)
    axes[0].set_title('Average Attractiveness by Motivation Factor\n(Top 8 Most Common)', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, 10)
    axes[0].axvline(driver_data['attractiveness'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f"Overall Mean: {driver_data['attractiveness'].mean():.2f}")
    axes[0].legend()
    
    for i, (idx, row) in enumerate(motivation_top.iterrows()):
        axes[0].text(row['mean'] + 0.1, i, f"{row['mean']:.2f}", va='center', fontweight='bold', fontsize=10)
    
    # Right: Box plot showing distribution
    motivation_list = motivation_top.index.tolist()
    data_to_plot = [driver_data[driver_data['motivation'] == m]['attractiveness'].values 
                    for m in motivation_list]
    
    bp = axes[1].boxplot(data_to_plot, tick_labels=motivation_list, vert=False, patch_artist=True,
                         showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors_gradient):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    axes[1].set_xlabel('Attractiveness Rating (1-10)', fontsize=11)
    axes[1].set_ylabel('Motivation Factor', fontsize=11)
    axes[1].set_title('Distribution of Attractiveness Ratings by Motivation Factor', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 10)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def chart_brand_perception_ladder(filtered_df: pd.DataFrame):
    """Brand Perception Ladder - Cell 4575"""
    if 'attractiveness' not in filtered_df.columns or 'perception' not in filtered_df.columns:
        return
    
    familiarity_data = filtered_df[(filtered_df['status'] == 'Complete') & 
                                  (filtered_df['attractiveness'].notna()) & 
                                  (filtered_df['perception'].notna())].copy()
    
    if len(familiarity_data) == 0:
        st.markdown("<p style='text-align: center; color: #666;'>No complete responses with perception data</p>", unsafe_allow_html=True)
        return
    
    perception_order = [
        "I don't know much about this organisation.",
        "I know about this organisation but have no opinion on it as an employer.",
        "I think this organisation would be a desirable place to work.",
        "I would apply for a position at this organisation."
    ]
    
    perception_means = familiarity_data.groupby('perception')['attractiveness'].mean()
    perception_means = perception_means.reindex(perception_order).dropna()
    
    if len(perception_means) == 0:
        st.markdown("<p style='text-align: center; color: #666;'>Insufficient perception data</p>", unsafe_allow_html=True)
        return
    
    # Create labels for perception levels
    perception_labels = []
    for perc in perception_means.index:
        if "don't know" in perc.lower():
            perception_labels.append("Don't Know Much\n(Awareness)")
        elif "no opinion" in perc.lower():
            perception_labels.append("Know, No Opinion\n(Recognition)")
        elif "desirable" in perc.lower():
            perception_labels.append("Desirable Employer\n(Consideration)")
        elif "would apply" in perc.lower():
            perception_labels.append("Would Apply\n(Intent)")
        else:
            perception_labels.append(perc[:20])
    
    colors_familiarity = ['#C0392B', '#E67E22', '#3498DB', '#27AE60'][:len(perception_means)]
    
    # Build data for violin plot - only for perception levels that have data
    data_violin = []
    perception_list_with_data = []
    for p in perception_order:
        data_subset = familiarity_data[familiarity_data['perception'] == p]['attractiveness'].values
        if len(data_subset) > 0:
            data_violin.append(data_subset)
            perception_list_with_data.append(p)
    
    # Create 2-subplot figure matching notebook
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    
    # Left: Bar chart with dividend arrows
    bars = axes[0].barh(range(len(perception_means)), perception_means.values, 
                       color=colors_familiarity, edgecolor='black', height=0.6)
    axes[0].set_yticks(range(len(perception_means)))
    axes[0].set_yticklabels(perception_labels, fontsize=10)
    axes[0].set_xlabel('Average Attractiveness Rating (1-10)', fontsize=11)
    axes[0].set_title('The Familiarity Ladder: Brand Awareness ROI', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, 10)
    
    # Add value labels and dividend arrows between levels
    for i, (bar, val) in enumerate(zip(bars, perception_means.values)):
        axes[0].text(val + 0.15, i, f"{val:.2f}", va='center', fontweight='bold', fontsize=11)
        
        # Add arrows showing dividend between levels
        if i > 0:
            prev_val = perception_means.values[i-1]
            dividend = val - prev_val
            mid_y = i - 0.5
            axes[0].annotate('', xy=(val, mid_y), xytext=(prev_val, mid_y),
                            arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
            axes[0].text((prev_val + val) / 2, mid_y + 0.15, f'+{dividend:.2f}', 
                        ha='center', fontsize=9, color='darkgreen', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Right: Violin plot showing distributions
    if len(data_violin) > 0:
        parts = axes[1].violinplot(data_violin, positions=range(len(data_violin)), 
                                   vert=False, widths=0.7, showmeans=True, showmedians=True)
        
        # Color the violins
        colors_for_violin = ['#C0392B', '#E67E22', '#3498DB', '#27AE60'][:len(data_violin)]
        for patch, color in zip(parts['bodies'], colors_for_violin):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        # Create labels for violin plot
        ytick_labels = []
        for p in perception_list_with_data:
            if "don't know" in p.lower():
                ytick_labels.append("Don't Know")
            elif "no opinion" in p.lower():
                ytick_labels.append("Know, No Opinion")
            elif "desirable" in p.lower():
                ytick_labels.append("Desirable")
            elif "would apply" in p.lower():
                ytick_labels.append("Would Apply")
            else:
                ytick_labels.append(p[:20])
        
        axes[1].set_yticks(range(len(perception_list_with_data)))
        axes[1].set_yticklabels(ytick_labels, fontsize=10)
        axes[1].set_xlabel('Attractiveness Rating (1-10)', fontsize=11)
        axes[1].set_title('Rating Distribution by Familiarity Level', fontsize=12, fontweight='bold')
        axes[1].set_xlim(0, 10)
    
    plt.tight_layout()
    st.pyplot(fig)


def chart_maturity_shift(filtered_df: pd.DataFrame):
    """Maturity Shift (Year 1 → 4) - Cell 3827"""
    maturity_data = filtered_df[(filtered_df['status'] == 'Complete') & 
                                 (filtered_df['year'].isin(['Year 1', 'Year 2', 'Year 3', 'Year 4']))].copy()
    
    if len(maturity_data) == 0:
        st.markdown("<p style='text-align: center; color: #666;'>No complete responses with year data</p>", unsafe_allow_html=True)
        return
    
    year_interests = {}
    for year in ['Year 1', 'Year 2', 'Year 3', 'Year 4']:
        year_df = maturity_data[maturity_data['year'] == year]
        year_interests[year] = {
            'Culture': (year_df['info_culture'].notna().sum() / len(year_df)) * 100,
            'Process': (year_df['info_process'].notna().sum() / len(year_df)) * 100,
            'Career': (year_df['info_career'].notna().sum() / len(year_df)) * 100,
            'Compensation': (year_df['info_comp'].notna().sum() / len(year_df)) * 100,
            'Roles': (year_df['info_roles'].notna().sum() / len(year_df)) * 100,
        }
    
    year_interest_df = pd.DataFrame(year_interests).T
    
    # Create 2-subplot figure matching notebook
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    key_categories = ['Culture', 'Process', 'Career', 'Compensation']
    for category in key_categories:
        axes[0].plot(year_interest_df.index, year_interest_df[category], 
                     marker='o', linewidth=2.5, markersize=8, label=category)
    
    axes[0].set_title('Evolution of Interests: Year 1 to Year 4', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Year of Study', fontsize=11)
    axes[0].set_ylabel('Interest Rate (%)', fontsize=11)
    axes[0].legend(title='Interest Category', loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Heatmap showing all interests by year
    sns.heatmap(year_interest_df.T, annot=True, fmt='.1f', cmap='RdYlGn', 
                ax=axes[1], cbar_kws={'label': 'Interest Rate (%)'})
    axes[1].set_title('Interest Heatmap: Academic Progression', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Year of Study', fontsize=11)
    axes[1].set_ylabel('Interest Category', fontsize=11)
    
    plt.tight_layout()
    st.pyplot(fig)


def render_placeholders():
    """Render placeholder cards for future visualizations."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>🔮 Future Visualizations</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='placeholder-card'>
            <h4>➕ Add New Visualization</h4>
            <p>Drag and drop or click to add</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='placeholder-card'>
            <h4>➕ Add New Visualization</h4>
            <p>Drag and drop or click to add</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# AI CHAT MODAL
# =============================================================================

def response_generator(prompt: str, df, response_key: str):
    """Stream AI response tokens."""
    response, code, image_path = run_analyst_agent(prompt, df)
    words = response.split()
    for word in words:
        yield word + " "
        time.sleep(0.02)
    
    if "pending_code" not in st.session_state:
        st.session_state.pending_code = {}
    if "pending_image" not in st.session_state:
        st.session_state.pending_image = {}
    
    st.session_state.pending_code[response_key] = code
    st.session_state.pending_image[response_key] = image_path


@st.dialog("🤖 AI Assistant")
def show_chat_modal():
    """Display the AI chat modal."""
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                if "image" in message and message["image"] and os.path.exists(message["image"]):
                    with open(message["image"], "rb") as f:
                        st.image(f.read())
                
                if "code" in message and message["code"]:
                    with st.expander("🛠️ View Generated Code"):
                        st.code(message["code"], language="python")
    
    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                import time as tm
                response_key = str(int(tm.time() * 1000))
                
                with st.spinner("Analyzing data..."):
                    response = st.write_stream(response_generator(prompt, filtered_df, response_key))
                    
                    code = st.session_state.pending_code.get(response_key)
                    image_path = st.session_state.pending_image.get(response_key)
                    
                    if image_path and os.path.exists(image_path):
                        with open(image_path, "rb") as f:
                            st.image(f.read())
                    
                    if code:
                        with st.expander("🛠️ View Generated Code"):
                            st.code(code, language="python")
                
                message_data = {"role": "assistant", "content": response, "code": code}
                if image_path:
                    message_data["image"] = image_path
                st.session_state.messages.append(message_data)
        st.rerun()


# =============================================================================
# MAIN APP
# =============================================================================

# Load data
df = load_data()
if df is None:
    st.error("Data file not found. Please ensure the Excel file is in the directory.")
    st.stop()

# Get filter options
all_schools = sorted(df['school'].dropna().unique()) if 'school' in df.columns else []
all_years = sorted(df['year'].dropna().unique()) if 'year' in df.columns else []

# Initialize session state
if 'school_selector' not in st.session_state:
    st.session_state.school_selector = all_schools
if 'year_selector' not in st.session_state:
    st.session_state.year_selector = all_years

# Render sidebar
gen_insights, selected_schools, selected_years = render_sidebar_filters()

# Apply filters
mask = pd.Series([True] * len(df))
if 'school' in df.columns:
    mask &= df['school'].isin(selected_schools)
if 'year' in df.columns:
    mask &= df['year'].isin(selected_years)
filtered_df = df[mask]

# Render sidebar metrics
render_sidebar_metrics(len(df), len(filtered_df))

# Generate insights
if gen_insights and not filtered_df.empty:
    with st.spinner("Generating insights for all graphs..."):
        llm, _ = build_analyst_agent(filtered_df)
        
        # Graph 1: Time vs Status
        if 'status' in filtered_df.columns and 'duration_sec' in filtered_df.columns:
            stats = filtered_df.groupby('status')['duration_sec'].describe().to_string()
            prompt = f"Analyze survey duration by status. Data:\n{stats}\nProvide 2 short, bulleted insights."
            st.session_state.insights['graph1'] = llm.invoke(prompt).content
        
        # Graph 2: Year vs Qualification
        if 'year' in filtered_df.columns and 'qualification' in filtered_df.columns:
            ct = pd.crosstab(filtered_df['year'], filtered_df['qualification'])
            if not ct.empty:
                prompt = f"Analyze Year vs Qualification. Data:\n{ct.to_string()}\nProvide 2 short, bulleted insights."
                st.session_state.insights['graph2'] = llm.invoke(prompt).content
        
        # Graph 3: Attractiveness
        if 'attractiveness' in filtered_df.columns:
            stats = filtered_df['attractiveness'].describe().to_string()
            prompt = f"Analyze attractiveness scores (1-10). Data:\n{stats}\nProvide 2 short, bulleted insights."
            st.session_state.insights['graph3'] = llm.invoke(prompt).content
        
        # Graph 4: Top Majors
        if 'subject' in filtered_df.columns:
            top = filtered_df['subject'].value_counts().head(10).to_string()
            prompt = f"Analyze top majors. Data:\n{top}\nProvide 2 short, bulleted insights."
            st.session_state.insights['graph4'] = llm.invoke(prompt).content

# Chat modal
if st.session_state.get("chat_open", False):
    show_chat_modal()

# Floating chat button
with st.container():
    col1, col2 = st.columns([0.9, 0.1])
    with col2:
        if st.button("💬", key="chat_btn", help="Open AI Assistant", type="primary"):
            st.session_state.chat_open = True
            st.rerun()

# Main content
render_header()
render_filter_bar(len(selected_schools), len(selected_years), len(filtered_df))

# Tabs
tab_quality, tab_results = st.tabs(["🛠️ Survey Quality", "📈 Survey Results"])

# Tab 1: Survey Quality
with tab_quality:
    render_metrics(filtered_df)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 1: Status Distribution & Drop-off Analysis
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        render_chart_card("📊 Status Distribution", lambda: chart_status_distribution(filtered_df))
    
    with row1_col2:
        render_chart_card("📊 Drop-off Analysis", lambda: chart_dropoff_analysis(filtered_df))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2: Question Correlation Heatmap & Partial Prediction Model
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        render_chart_card("📊 Question Correlation Heatmap", lambda: chart_question_correlation_heatmap(filtered_df))
    
    with row2_col2:
        render_chart_card("📊 Partial Prediction Model Performance", lambda: chart_partial_prediction_model(filtered_df))

# Tab 2: Survey Results
with tab_results:
    # Row 1: Interest-Based Personas & Motivation Drivers
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        render_chart_card("📊 Interest-Based Personas", lambda: chart_interest_personas(filtered_df))
    
    with row3_col2:
        render_chart_card("📊 Motivation Drivers", lambda: chart_motivation_drivers(filtered_df))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2: Brand Perception Ladder & Maturity Shift
    row4_col1, row4_col2 = st.columns(2)
    
    with row4_col1:
        render_chart_card("📊 Brand Perception Ladder", lambda: chart_brand_perception_ladder(filtered_df))
    
    with row4_col2:
        render_chart_card("📊 Maturity Shift (Year 1 → 4)", lambda: chart_maturity_shift(filtered_df))
