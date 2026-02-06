"""
GradSingapore Survey Analytics Dashboard
======================================
A professional Streamlit dashboard for analyzing student survey data.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from langchain_openai import ChatOpenAI
import os
import uuid
import time
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    /* --- Chart Container --- */
    .chart-container {
        margin-bottom: 12px;
    }
    
    /* --- Insight Text (scrollable) --- */
    .insight-scroll {
        font-size: 14px;
        color: #374151;
        margin-top: 12px;
        padding: 16px;
        border-top: 2px solid #e5e7eb;
        line-height: 1.6;
        max-height: 200px;
        overflow-y: auto;
        background: #f9fafb;
        border-radius: 8px;
    }
    .insight-scroll strong {
        color: #2563eb;
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
- Answer user questions about the survey data
- Provide text explanations, statistics, and insights
- Only generate Python code and visualizations if the user EXPLICITLY asks for a chart, graph, or visualization

OUTPUT RULES:
- If user asks a question: respond with clear text explanations and statistics
- If user asks for a visualization (e.g., "show me", "graph", "chart", "plot", "visualize"):
    - Output ONLY valid Python code
    - Do NOT include explanations outside code
    - Do NOT use markdown
    - call plt.clf() before plotting
    - save the figure to 'temp_plot.png'
    - do NOT call plt.show()
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
    
    # Check if user is asking for a visualization
    viz_keywords = ['show', 'chart', 'graph', 'plot', 'visualize', 'display', 'draw', 'create a', 'make a']
    is_visualization_request = any(keyword in user_question.lower() for keyword in viz_keywords)
    
    if is_visualization_request:
        prompt = f"{system_prompt}\n\nUser question:\n{user_question}"
    else:
        # For text-only questions, modify prompt to not expect code
        text_prompt = system_prompt.replace("\nOUTPUT RULES:", "\nOUTPUT RULES:\n- For text questions: respond directly with explanations and statistics\n- For text questions: Do NOT generate code")
        prompt = f"{text_prompt}\n\nUser question:\n{user_question}\n\nRespond with clear text explanation only (no code needed)."
    
    try:
        response = llm.invoke(prompt)
        content = response.content

        # Check if response contains code (```python or plt./sns./plot)
        if "```python" in content or ("plt." in content and ("savefig" in content or "save('" in content or 'save("' in content or ".png" in content or ".jpg" in content)):
            # Extract code
            code = content
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            code = code.strip()

            local_vars = {"df": df.copy(), "pd": pd, "plt": plt, "sns": sns}
            exec(code, {}, local_vars)

            # Handle generated plot
            image_path = None
            if os.path.exists("temp_plot.png"):
                unique_filename = f"chart_{uuid.uuid4().hex}.png"
                os.rename("temp_plot.png", unique_filename)
                image_path = unique_filename

            return code, image_path
        else:
            # No visualization code - return as plain text
            # Clean up any markdown formatting
            clean_text = content.replace("```", "").strip()
            return clean_text, None

    except Exception as e:
        return f"Error: {str(e)}", None


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


def render_chart_card(title: str, chart_function, insight_key: str | None = None, card_height: int = 800):
    """Helper to render a chart card with optional insight."""
    st.markdown(f"**{title}**")
    chart_function()
    # Scrollable insights using Streamlit expander
    if insight_key and insight_key in st.session_state.insights:
        with st.expander("💡 View Insights", expanded=True):
            st.markdown(st.session_state.insights[insight_key])
    else:
        st.info("💡 Click '✨ Generate Insights' in sidebar to see AI-generated insights")


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
    
    fig = px.pie(
        names=status_counts.index,
        values=status_counts.values,
        hole=0.5,
        color=status_counts.index,
        color_discrete_map=status_colors,
        title='Survey Status Distribution'
    )
    
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        legend_title_text='Status',
        font=dict(size=12),
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    st.plotly_chart(fig, width='stretch')


def chart_dropoff_analysis(filtered_df: pd.DataFrame):
    """Drop-off Analysis (Bar Chart) - Cell 3432"""
    if 'status' not in filtered_df.columns:
        return
    
    survey_questions = [
        'school', 'year', 'qualification', 'subject', 'nationality',
        'gender', 'perception', 'info_roles', 'info_career', 'info_comp',
        'info_culture', 'info_process', 'info_other_text', 'attractiveness',
        'motivation', 'motivation_other'
    ]
    
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
    
    # Create readable labels
    question_labels = {
        'school': 'School',
        'year': 'Year of Study',
        'qualification': 'Qualification',
        'subject': 'Subject/Major',
        'nationality': 'Nationality',
        'gender': 'Gender',
        'perception': 'Employer Perception',
        'info_roles': 'Types of Roles',
        'info_career': 'Career Progression',
        'info_comp': 'Compensation & Benefits',
        'info_culture': 'Work-Life Balance & Culture',
        'info_process': 'Application Process',
        'info_other_text': 'Other Info Interest',
        'attractiveness': 'Attractiveness Rating',
        'motivation': 'Motivation Factor',
        'motivation_other': 'Other Motivation'
    }
    dropoff_summary['question_label'] = dropoff_summary['question'].map(
        lambda x: question_labels.get(x, x)
    )
    
    fig = px.bar(
        dropoff_summary,
        x='num_dropoffs',
        y='question_label',
        orientation='h',
        title='First Drop-Off Point for Partial Survey Responses',
        labels={'question_label': 'Question Where Respondents First Stopped', 'num_dropoffs': 'Number of Respondents'},
        text='num_dropoffs',
        color='num_dropoffs',
        color_continuous_scale='Reds'
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Drop-offs: %{x}<br>Percentage: %{customdata:.1f}%<extra></extra>',
        customdata=dropoff_summary['percentage'].values
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False,
        font=dict(size=12),
        margin=dict(t=50, b=50, l=200, r=20)
    )
    
    st.plotly_chart(fig, width='stretch')


def chart_question_correlation_heatmap(filtered_df: pd.DataFrame):
    """Question Correlation Heatmap - Cell 2798"""
    info_cols = ['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']
    
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
    
    info_corr_renamed = info_corr.rename(index=info_rename_map, columns=info_rename_map)
    
    fig = px.imshow(
        info_corr_renamed,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Redundancy Check: Learn More Correlations'
    )
    
    fig.update_layout(
        font=dict(size=11),
        margin=dict(t=50, b=50, l=20, r=20)
    )
    
    st.plotly_chart(fig, width='stretch')


def chart_partial_prediction_model(filtered_df: pd.DataFrame):
    """Factors Driving Partial Survey Responses - Feature Importance"""
    model_path = 'partial_response_model.pkl'
    encoders_path = 'partial_response_label_encoders.pkl'
    features_path = 'partial_response_feature_list.pkl'
    
    if not (os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(features_path)):
        st.markdown("<p style='text-align: center; color: #666;'>Model files not found</p>", unsafe_allow_html=True)
        return
    
    try:
        import joblib
        from sklearn.preprocessing import LabelEncoder
        
        rf = joblib.load(model_path)
        label_encoders = joblib.load(encoders_path)
        analysis_vars = joblib.load(features_path)
        
        df_work = filtered_df.copy()
        if 'school' in df_work.columns and 'university' not in df_work.columns:
            df_work = df_work.rename(columns={'school': 'university'})
        
        analysis_vars_local = [v for v in analysis_vars if v in df_work.columns]
        
        missing_features = [v for v in analysis_vars if v not in df_work.columns]
        if missing_features:
            st.markdown(f"<p style='text-align: center; color: #666;'>Missing features: {', '.join(missing_features)}</p>", unsafe_allow_html=True)
            return
        
        df_model = df_work[analysis_vars + ['status']].dropna().copy()
        
        if len(df_model) < len(analysis_vars) + 1:
            st.markdown("<p style='text-align: center; color: #666;'>Insufficient data</p>", unsafe_allow_html=True)
            return
        
        df_model['is_partial'] = (df_model['status'] == 'Partial').astype(int)
        
        X = df_model[analysis_vars].copy()
        for col in analysis_vars:
            if col in label_encoders:
                le = label_encoders[col]
                X[col] = X[col].astype(str)
                X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        
        y = df_model['is_partial']
        
        if len(y.unique()) < 2:
            st.markdown("<p style='text-align: center; color: #666;'>Insufficient class variation</p>", unsafe_allow_html=True)
            return
        
        importances = pd.Series(rf.feature_importances_, index=analysis_vars).sort_values(ascending=True)
        
        # Create readable feature names
        feature_labels = {
            'school': 'School',
            'year': 'Year of Study',
            'qualification': 'Qualification',
            'subject': 'Subject/Major',
            'nationality': 'Nationality',
            'gender': 'Gender',
            'perception': 'Employer Perception',
            'attractiveness': 'Attractiveness Rating'
        }
        
        importances.index = [feature_labels.get(x, x) for x in importances.index]
        
        fig = px.bar(
            x=importances.values,
            y=importances.index,
            orientation='h',
            title='Predictive Importance: Factors Driving Partial Responses',
            labels={'x': 'Importance Score', 'y': 'Feature'},
            text_auto='.3f',
            color=importances.values,
            color_continuous_scale='Oranges'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        )
        
        fig.update_layout(
            coloraxis_showscale=False,
            font=dict(size=11),
            margin=dict(t=50, b=50, l=150, r=20)
        )
        
        st.plotly_chart(fig, width='stretch')
        
    except Exception as e:
        st.markdown(f"<p style='text-align: center; color: #666;'>Model visualization unavailable</p>", unsafe_allow_html=True)


# =============================================================================
# TAB 2: SURVEY RESULTS GRAPHS
# =============================================================================

def _get_clustering_data(filtered_df: pd.DataFrame):
    """Helper function to perform clustering and return processed data."""
    info_cols = ['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']
    
    missing_cols = [col for col in info_cols if col not in filtered_df.columns]
    if missing_cols or 'subject' not in filtered_df.columns:
        return None, "Missing required columns"
    
    clustering_data = filtered_df[info_cols + ['subject', 'attractiveness']].copy()
    clustering_data[info_cols] = clustering_data[info_cols].notnull().astype(int)
    clustering_clean = clustering_data.dropna(subset=['subject', 'attractiveness']).copy()
    
    if len(clustering_clean) < 3:
        return None, "Insufficient data for clustering"
    
    X_cluster = clustering_clean[info_cols].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clustering_clean.loc[:, 'persona'] = kmeans.fit_predict(X_cluster)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=info_cols)
    
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
    
    return clustering_clean, cluster_centers, persona_labels, interest_names


def plot_interest_personas_heatmap(filtered_df: pd.DataFrame):
    """Interest-Based Personas: Cluster Profiles Heatmap"""
    result = _get_clustering_data(filtered_df)
    if result[0] is None:
        st.markdown(f"<p style='text-align: center; color: #666;'>{result[1]}</p>", unsafe_allow_html=True)
        return
    
    clustering_clean, cluster_centers, persona_labels, interest_names = result
    
    # Prepare data for heatmap
    cluster_centers_display = cluster_centers.copy()
    cluster_centers_display.columns = [interest_names.get(c, c) for c in cluster_centers_display.columns]
    cluster_centers_display.index = persona_labels
    
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=cluster_centers_display.values,
            x=cluster_centers_display.columns,
            y=cluster_centers_display.index,
            colorscale='YlOrRd',
            text=np.round(cluster_centers_display.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=12, color='black'),
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>',
            colorbar=dict(title='Interest Intensity')
        )
    )
    
    fig.update_layout(
        title_text='Interest-Based Personas: Cluster Profiles',
        height=450,
        xaxis_title='Interest Categories',
        yaxis_title='Persona Type',
        showlegend=False,
        font=dict(size=11),
        margin=dict(t=60, b=50, l=20, r=20)
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_interest_personas_distribution(filtered_df: pd.DataFrame):
    """Interest-Based Personas: Distribution Bar Chart"""
    result = _get_clustering_data(filtered_df)
    if result[0] is None:
        st.markdown(f"<p style='text-align: center; color: #666;'>{result[1]}</p>", unsafe_allow_html=True)
        return
    
    clustering_clean, _, _, _ = result
    
    # Prepare data for bar chart
    persona_counts = clustering_clean['persona_label'].value_counts().reset_index()
    persona_counts.columns = ['Persona', 'Count']
    
    colors_persona = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=persona_counts['Count'],
            y=persona_counts['Persona'],
            orientation='h',
            marker_color=colors_persona[:len(persona_counts)],
            text=persona_counts['Count'],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title_text='Distribution of Interest-Based Personas',
        height=450,
        xaxis_title='Number of Respondents',
        yaxis_title='Persona Type',
        showlegend=False,
        font=dict(size=11),
        margin=dict(t=60, b=50, l=20, r=20)
    )
    
    st.plotly_chart(fig, width='stretch')


def chart_interest_personas(filtered_df: pd.DataFrame):
    """Interest-Based Personas (Clustering) - Cell 3590"""
    # This function is kept for backward compatibility
    # New separate functions: plot_interest_personas_heatmap, plot_interest_personas_distribution
    col1, col2 = st.columns(2)
    with col1:
        plot_interest_personas_heatmap(filtered_df)
    with col2:
        plot_interest_personas_distribution(filtered_df)


def _get_motivation_data(filtered_df: pd.DataFrame):
    """Helper function to prepare motivation data."""
    if 'attractiveness' not in filtered_df.columns or 'motivation' not in filtered_df.columns:
        return None, "Missing required columns"
    
    driver_data = filtered_df[(filtered_df['status'] == 'Complete') & 
                             (filtered_df['attractiveness'].notna()) & 
                             (filtered_df['motivation'].notna())].copy()
    
    if len(driver_data) == 0:
        return None, "No complete responses with motivation data"
    
    motivation_attractiveness = driver_data.groupby('motivation')['attractiveness'].agg(['mean', 'count']).round(2)
    motivation_attractiveness = motivation_attractiveness.sort_values('mean', ascending=True)
    
    top_motivations = motivation_attractiveness.nlargest(8, 'count').index
    motivation_top = motivation_attractiveness[motivation_attractiveness.index.isin(top_motivations)]
    
    return driver_data, motivation_top


def plot_motivation_bar(filtered_df: pd.DataFrame):
    """Motivation Drivers: Average Attractiveness by Factor (Bar Chart)"""
    result = _get_motivation_data(filtered_df)
    if result[0] is None:
        st.markdown(f"<p style='text-align: center; color: #666;'>{result[1]}</p>", unsafe_allow_html=True)
        return
    
    driver_data, motivation_top = result
    
    colors_gradient = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027'][:len(motivation_top)]
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=motivation_top.index,
            x=motivation_top['mean'],
            orientation='h',
            marker_color=colors_gradient,
            text=np.round(motivation_top['mean'], 2),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Avg Attractiveness: %{x:.2f}<br>Count: %{customdata}<extra>',
            customdata=motivation_top['count'].values
        )
    )
    
    # Add vertical line for overall mean
    overall_mean = driver_data['attractiveness'].mean()
    fig.add_vline(x=overall_mean, line_dash='dash', line_color='red', line_width=2)
    fig.add_annotation(x=overall_mean, y=len(motivation_top), text=f'Overall Mean: {overall_mean:.2f}',
                      showarrow=True, arrowhead=2, yshift=10)
    
    fig.update_layout(
        title_text='Average Attractiveness by Motivation Factor (Top 8)',
        height=450,
        xaxis_title='Average Attractiveness Rating (1-10)',
        yaxis_title='Motivation Factor',
        showlegend=False,
        font=dict(size=10),
        margin=dict(t=60, b=50, l=20, r=20),
        xaxis=dict(range=[0, 10])
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_motivation_box(filtered_df: pd.DataFrame):
    """Motivation Drivers: Distribution of Ratings (Box Plot)"""
    result = _get_motivation_data(filtered_df)
    if result[0] is None:
        st.markdown(f"<p style='text-align: center; color: #666;'>{result[1]}</p>", unsafe_allow_html=True)
        return
    
    driver_data, motivation_top = result
    
    colors_gradient = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027'][:len(motivation_top)]
    motivation_list = motivation_top.index.tolist()
    
    fig = go.Figure()
    
    for i, motivation in enumerate(motivation_list):
        data = driver_data[driver_data['motivation'] == motivation]['attractiveness'].values
        fig.add_trace(
            go.Box(
                x=data,
                name=motivation,
                orientation='h',
                marker_color=colors_gradient[i % len(colors_gradient)],
                boxmean=True,
                hovertemplate='<b>%{name}</b><br>Min: %{min}<br>Q1: %{q1}<br>Median: %{median}<br>Q3: %{q3}<br>Max: %{max}<br>Mean: %{mean:.2f}<extra>'
            )
        )
    
    fig.update_layout(
        title_text='Distribution of Attractiveness Ratings',
        height=450,
        xaxis_title='Attractiveness Rating (1-10)',
        yaxis_title='Motivation Factor',
        showlegend=False,
        font=dict(size=10),
        margin=dict(t=60, b=50, l=20, r=20),
        xaxis=dict(range=[0, 10])
    )
    
    st.plotly_chart(fig, width='stretch')


def chart_motivation_drivers(filtered_df: pd.DataFrame):
    """Motivation Drivers (Box Plot) - Cell 4481"""
    # This function is kept for backward compatibility
    # New separate functions: plot_motivation_bar, plot_motivation_box
    col1, col2 = st.columns(2)
    with col1:
        plot_motivation_bar(filtered_df)
    with col2:
        plot_motivation_box(filtered_df)


def _get_gap_data(filtered_df: pd.DataFrame):
    """Helper function to prepare gap analysis data."""
    gap_data = filtered_df[filtered_df['status'] == 'Complete'].copy()
    
    if len(gap_data) == 0:
        return None, "No complete responses to analyze"
    
    info_cols = ['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']
    
    missing_cols = [col for col in info_cols if col not in gap_data.columns]
    if missing_cols:
        return None, f"Missing columns: {', '.join(missing_cols)}"
    
    info_names = {
        'info_roles': 'Types of Roles',
        'info_career': 'Career Progression',
        'info_comp': 'Compensation & Benefits',
        'info_culture': 'Work-Life Balance & Culture',
        'info_process': 'Application Process'
    }
    
    info_interest_rates = {}
    for col in info_cols:
        rate = (gap_data[col].notna().sum() / len(gap_data)) * 100
        info_interest_rates[info_names[col]] = rate
    
    if 'motivation' not in gap_data.columns:
        return None, "Missing motivation column"
    
    motivation_rates = (gap_data['motivation'].value_counts() / len(gap_data)) * 100
    
    motivation_mapping = {
        'Compensation & Benefits': ['Competitive salary', 'Compensation and benefits', 
                                    'Sign-on bonus', 'Performance bonus'],
        'Career Progression': ['Career development and growth opportunities', 
                              'Career progression', 'Professional development'],
        'Work-Life Balance & Culture': ['Work-life balance', 'Company culture', 
                                        'Flexible working arrangements', 'Team environment'],
        'Types of Roles': ['Challenging and interesting work', 'Job security', 
                          'Variety of roles', 'Role diversity'],
        'Application Process': ['Recruitment process', 'Interview process']
    }
    
    aggregated_motivation = {}
    for category, keywords in motivation_mapping.items():
        total_rate = 0
        for keyword in keywords:
            matches = motivation_rates[motivation_rates.index.str.contains(keyword, case=False, na=False)]
            total_rate += matches.sum()
        aggregated_motivation[category] = total_rate
    
    gap_df = pd.DataFrame({
        'Want to Learn (%)': pd.Series(info_interest_rates),
        'Motivates to Apply (%)': pd.Series(aggregated_motivation)
    }).fillna(0)
    
    gap_df['Gap'] = gap_df['Want to Learn (%)'] - gap_df['Motivates to Apply (%)']
    gap_df = gap_df.sort_values('Gap', ascending=False)
    
    if len(gap_df) == 0:
        return None, "Insufficient data for gap analysis"
    
    return gap_df


def plot_information_gap_grouped(filtered_df: pd.DataFrame):
    """Information Gap: Grouped Bar Chart (Side-by-Side)"""
    gap_df = _get_gap_data(filtered_df)
    if gap_df is None:
        st.markdown("<p style='text-align: center; color: #666;'>Insufficient data for gap analysis</p>", unsafe_allow_html=True)
        return
    
    categories = gap_df.index.tolist()
    x_pos = np.arange(len(categories))
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            name='Want to Learn About',
            x=x_pos - 0.175,
            y=gap_df['Want to Learn (%)'],
            width=0.35,
            marker_color='#3498DB',
            text=np.round(gap_df['Want to Learn (%)'], 1),
            textposition='outside',
            hovertemplate='<b>%{customdata}</b><br>Want to Learn: %{y:.1f}%<extra>',
            customdata=categories
        )
    )
    
    fig.add_trace(
        go.Bar(
            name='Motivates to Apply',
            x=x_pos + 0.175,
            y=gap_df['Motivates to Apply (%)'],
            width=0.35,
            marker_color='#E74C3C',
            text=np.round(gap_df['Motivates to Apply (%)'], 1),
            textposition='outside',
            hovertemplate='<b>%{customdata}</b><br>Motivates: %{y:.1f}%<extra>',
            customdata=categories
        )
    )
    
    fig.update_layout(
        title_text='Information Curiosity vs Motivation Gap',
        height=450,
        xaxis_title='Category',
        yaxis_title='Percentage (%)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        font=dict(size=10),
        margin=dict(t=60, b=80, l=20, r=20),
        xaxis=dict(tickvals=x_pos, ticktext=categories, tickangle=30, tickfont=dict(size=9))
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_information_gap_lollipop(filtered_df: pd.DataFrame):
    """Information Gap: Lollipop Chart (Gap Visualization)"""
    gap_df = _get_gap_data(filtered_df)
    if gap_df is None:
        st.markdown("<p style='text-align: center; color: #666;'>Insufficient data for gap analysis</p>", unsafe_allow_html=True)
        return
    
    gap_sorted = gap_df.sort_values('Gap')
    colors_gap = ['#27AE60' if x < 0 else '#E74C3C' for x in gap_sorted['Gap']]
    y_pos = np.arange(len(gap_sorted))
    
    fig = go.Figure()
    
    # Add horizontal lines for lollipop effect
    for i, (idx, row) in enumerate(gap_sorted.iterrows()):
        fig.add_trace(
            go.Scatter(
                x=[0, row['Gap']],
                y=[i, i],
                mode='lines',
                line=dict(color=colors_gap[i], width=3),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=gap_sorted['Gap'],
            y=y_pos,
            mode='markers+text',
            marker=dict(size=14, color=colors_gap, line=dict(color='black', width=1.5)),
            text=[f'{v:+.1f}pp' for v in gap_sorted['Gap']],
            textfont=dict(size=10, color=colors_gap),
            hovertemplate='<b>%{customdata}</b><br>Gap: %{x:+.1f}pp<extra>',
            customdata=gap_sorted.index
        )
    )
    
    # Add vertical line at 0
    fig.add_vline(x=0, line_color='black', line_width=1)
    
    # Add annotations
    fig.add_annotation(x=0.98, y=0.02, xref='paper', yref='paper', text='← More Motivation',
                      showarrow=False, font=dict(size=10, color='#27AE60'),
                      xanchor='right', yanchor='bottom')
    fig.add_annotation(x=0.02, y=0.98, xref='paper', yref='paper', text='More Curiosity →',
                      showarrow=False, font=dict(size=10, color='#E74C3C'),
                      xanchor='left', yanchor='top')
    
    fig.update_layout(
        title_text='Gap Analysis (Positive = More Curious Than Motivated)',
        height=450,
        xaxis_title='Gap (Curiosity - Motivation) in pp',
        yaxis_title='',
        showlegend=False,
        font=dict(size=10),
        margin=dict(t=60, b=80, l=20, r=20),
        xaxis=dict(range=[-50, 80]),
        yaxis=dict(tickvals=y_pos, ticktext=gap_sorted.index, tickfont=dict(size=9))
    )
    
    st.plotly_chart(fig, width='stretch')


def chart_information_gap_analysis(filtered_df: pd.DataFrame):
    """Information Gap Analysis - Cell 4775"""
    # This function is kept for backward compatibility
    # New separate functions: plot_information_gap_grouped, plot_information_gap_lollipop
    col1, col2 = st.columns(2)
    with col1:
        plot_information_gap_grouped(filtered_df)
    with col2:
        plot_information_gap_lollipop(filtered_df)


def _get_maturity_data(filtered_df: pd.DataFrame):
    """Helper function to prepare maturity shift data."""
    maturity_data = filtered_df[(filtered_df['status'] == 'Complete') & 
                                 (filtered_df['year'].isin(['Year 1', 'Year 2', 'Year 3', 'Year 4']))].copy()
    
    if len(maturity_data) == 0:
        return None, "No complete responses with year data"
    
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
    return year_interest_df


def plot_maturity_line(filtered_df: pd.DataFrame):
    """Maturity Shift: Line Chart (Evolution of Interests)"""
    year_interest_df = _get_maturity_data(filtered_df)
    if year_interest_df is None:
        st.markdown("<p style='text-align: center; color: #666;'>No complete responses with year data</p>", unsafe_allow_html=True)
        return
    
    key_categories = ['Culture', 'Process', 'Career', 'Compensation']
    colors_lines = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']
    
    fig = go.Figure()
    
    for i, category in enumerate(key_categories):
        fig.add_trace(
            go.Scatter(
                x=year_interest_df.index,
                y=year_interest_df[category],
                mode='lines+markers',
                name=category,
                line=dict(width=3),
                marker=dict(size=10),
                hovertemplate=f'<b>{category}</b><br>Year: %{{x}}<br>Interest: %{{y:.1f}}%<extra>'
            )
        )
    
    fig.update_layout(
        title_text='Evolution of Interests: Year 1 to Year 4',
        height=450,
        xaxis_title='Year of Study',
        yaxis_title='Interest Rate (%)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        font=dict(size=11),
        margin=dict(t=60, b=50, l=20, r=20)
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_maturity_heatmap(filtered_df: pd.DataFrame):
    """Maturity Shift: Heatmap (Academic Progression)"""
    year_interest_df = _get_maturity_data(filtered_df)
    if year_interest_df is None:
        st.markdown("<p style='text-align: center; color: #666;'>No complete responses with year data</p>", unsafe_allow_html=True)
        return
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Heatmap(
            z=year_interest_df.T.values,
            x=year_interest_df.index,
            y=year_interest_df.columns,
            colorscale='RdYlGn',
            text=np.round(year_interest_df.T.values, 1),
            texttemplate='%{text}',
            textfont=dict(size=11, color='black'),
            hovertemplate='<b>%{y}</b><br>Year: %{x}<br>Interest: %{z:.1f}%<extra>',
            colorbar=dict(title='Interest Rate (%)')
        )
    )
    
    fig.update_layout(
        title_text='Interest Heatmap: Academic Progression',
        height=450,
        xaxis_title='Year of Study',
        yaxis_title='Interest Category',
        showlegend=False,
        font=dict(size=11),
        margin=dict(t=60, b=50, l=20, r=20)
    )
    
    st.plotly_chart(fig, width='stretch')


def chart_maturity_shift(filtered_df: pd.DataFrame):
    """Maturity Shift (Year 1 → 4) - Cell 3827"""
    # This function is kept for backward compatibility
    # New separate functions: plot_maturity_line, plot_maturity_heatmap
    col1, col2 = st.columns(2)
    with col1:
        plot_maturity_line(filtered_df)
    with col2:
        plot_maturity_heatmap(filtered_df)


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
    code, image_path = run_analyst_agent(prompt, df)
    
    # Check if response is an error or code
    if "error" in code.lower() or code.startswith("Error:"):
        response = code
        code = None
    else:
        # Check if code block markers exist (indicates visualization code)
        if "```python" in code or "plt." in code or "sns." in code or "plot(" in code:
            response = "Here is the visualization based on your request:"
        else:
            # No visualization code - return as plain text response
            response = code
            code = None
            image_path = None
    
    # Stream the response
    words = response.split()
    for word in words:
        yield word + " "
        time.sleep(0.02)
    
    # Store code and image for display after streaming
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

# Get filter options (include Unknown for NaN values)
all_schools = list(df['school'].fillna('Unknown').unique())
all_schools = sorted([s for s in all_schools if pd.notna(s) or s == 'Unknown'])
all_years = list(df['year'].fillna('Unknown').unique())
all_years = sorted([y for y in all_years if pd.notna(y) or y == 'Unknown'])

# Replace 'Unknown' back to NaN in df for filtering
df['school'] = df['school'].fillna('Unknown')
df['year'] = df['year'].fillna('Unknown')

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
        
        # Filter context for all prompts
        filter_context = f"""
FILTERS APPLIED:
- Schools: {', '.join(selected_schools) if selected_schools else 'All'}
- Years: {', '.join(selected_years) if selected_years else 'All'}
- Total responses analyzed: {len(filtered_df)}
"""
        
        # 1. Status Distribution
        if 'status' in filtered_df.columns:
            status_counts = filtered_df['status'].value_counts().to_string()
            status_pct = (filtered_df['status'].value_counts(normalize=True) * 100).round(1).to_string()
            completion_rate = (filtered_df['status'] == 'Complete').mean() * 100
            partial_rate = (filtered_df['status'] == 'Partial').mean() * 100
            disqualify_rate = (filtered_df['status'] == 'Disqualified').mean() * 100
            avg_duration = filtered_df['duration_sec'].mean() if 'duration_sec' in filtered_df.columns else None
            
            data_context = f"""
{filter_context}
STATUS DISTRIBUTION:
Counts:\n{status_counts}
Percentages:\n{status_pct}

KEY METRICS:
- Completion Rate: {completion_rate:.1f}%
- Partial Rate: {partial_rate:.1f}%
- Disqualification Rate: {disqualify_rate:.1f}%
- Average Duration: {avg_duration:.1f} seconds""" if avg_duration else f"""
{filter_context}
STATUS DISTRIBUTION:
Counts:\n{status_counts}
Percentages:\n{status_pct}

KEY METRICS:
- Completion Rate: {completion_rate:.1f}%
- Partial Rate: {partial_rate:.1f}%
- Disqualification Rate: {disqualify_rate:.1f}%
"""
            
            prompt = f"""Analyze this survey status data comprehensively:
{data_context}

Provide 3 actionable insights with specific recommendations for:
1. What the completion rate suggests about survey engagement
2. How to reduce partial/drop-off rates
3. Any patterns that need immediate attention"""
            st.session_state.insights['status'] = llm.invoke(prompt).content
        
        # 2. Drop-off Analysis
        if 'status' in filtered_df.columns:
            partial_df = filtered_df[filtered_df['status'] == 'Partial']
            dropoff_count = len(partial_df)
            completion_count = (filtered_df['status'] == 'Complete').sum()
            
            # Get first blank question for partials
            survey_questions = ['school', 'year', 'qualification', 'subject', 'nationality', 'gender', 
                              'perception', 'info_roles', 'info_career', 'info_comp', 'info_culture', 
                              'info_process', 'info_other_text', 'attractiveness', 'motivation', 'motivation_other']
            survey_questions = [q for q in survey_questions if q in filtered_df.columns]
            
            def first_blank(row):
                for q in survey_questions:
                    if pd.isna(row[q]) or str(row[q]).strip() == "":
                        return q
                return None
            
            if len(partial_df) > 0:
                partial_df_work = partial_df.copy()
                partial_df_work['first_dropoff'] = partial_df_work.apply(first_blank, axis=1)
                dropoff_summary = partial_df_work['first_dropoff'].value_counts().head(5).to_string()
            else:
                dropoff_summary = "No partial responses"
            
            data_context = f"""
{filter_context}
DROP-OFF ANALYSIS:
- Total Responses: {len(filtered_df)}
- Complete: {completion_count} ({completion_count/len(filtered_df)*100:.1f}%)
- Partial/Drop-off: {dropoff_count} ({dropoff_count/len(filtered_df)*100:.1f}%)

TOP DROP-OFF POINTS (first blank question):\n{dropoff_summary}
"""
            
            prompt = f"""Analyze this survey drop-off data:
{data_context}

Provide 3 actionable insights:
1. Which question is causing the most drop-offs and why
2. How to redesign the survey to improve completion
3. Targeted recommendations to reduce partial response rate"""
            st.session_state.insights['dropoff'] = llm.invoke(prompt).content
        
        # 3. Question Correlation Heatmap
        if all(col in filtered_df.columns for col in ['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']):
            info_binary = filtered_df[['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']].notna().astype(int)
            corr = info_binary.corr().round(2).to_string()
            
            # Get interest rates
            interest_rates = (info_binary.sum() / len(info_binary) * 100).round(1).to_string()
            
            data_context = f"""
{filter_context}
CORRELATION MATRIX:\n{corr}

INTEREST RATES (% who selected each option):\n{interest_rates}
"""
            
            prompt = f"""Analyze this correlation data from 'Learn More' questions:
{data_context}

Provide 3 actionable insights:
1. Which information topics are redundant (highly correlated)?
2. What unique interests should be prioritized?
3. Recommendations for consolidating survey questions"""
            st.session_state.insights['correlation'] = llm.invoke(prompt).content
        
        # 4. Partial Response Drivers
        if 'status' in filtered_df.columns:
            # Get model data if available
            import os
            if os.path.exists('partial_response_model.pkl'):
                import joblib
                rf = joblib.load('partial_response_model.pkl')
                analysis_vars = joblib.load('partial_response_feature_list.pkl')
                
                # Calculate partial rate by demographic
                partial_rate_by_school = (filtered_df.groupby('school')['status'].apply(lambda x: (x == 'Partial').mean() * 100)).round(1).to_string()
                partial_rate_by_year = (filtered_df.groupby('year')['status'].apply(lambda x: (x == 'Partial').mean() * 100)).round(1).to_string()
                partial_rate_by_qual = (filtered_df.groupby('qualification')['status'].apply(lambda x: (x == 'Partial').mean() * 100)).round(1).to_string()
                
                # Feature importance
                importance = pd.Series(rf.feature_importances_, index=analysis_vars).sort_values(ascending=False).round(3).to_string()
                
                data_context = f"""
{filter_context}
PARTIAL RESPONSE DRIVERS:
Feature Importance:\n{importance}

Partial Rate by School:\n{partial_rate_by_school}

Partial Rate by Year:\n{partial_rate_by_year}

Partial Rate by Qualification:\n{partial_rate_by_qual}
"""
            else:
                partial_by_school = (filtered_df.groupby('school')['status'].value_counts(normalize=True).unstack() * 100).round(1).to_string()
                partial_by_year = (filtered_df.groupby('year')['status'].value_counts(normalize=True).unstack() * 100).round(1).to_string()
                
                data_context = f"""
{filter_context}
PARTIAL RESPONSE ANALYSIS:

Partial Rate by School:\n{partial_by_school}

Partial Rate by Year:\n{partial_by_year}
"""
            
            prompt = f"""Analyze what drives partial survey responses:
{data_context}

Provide 3 actionable insights:
1. Which demographics have highest partial rate?
2. Why might certain groups be dropping off?
3. Targeted interventions to improve completion for at-risk groups"""
            st.session_state.insights['partial_drivers'] = llm.invoke(prompt).content
        
        # 5. Interest-Based Personas
        if 'subject' in filtered_df.columns and 'attractiveness' in filtered_df.columns:
            subject_counts = filtered_df['subject'].value_counts().head(15).to_string()
            avg_attract_by_subject = filtered_df.groupby('subject')['attractiveness'].mean().round(2).sort_values(ascending=False).head(15).to_string()
            
            # Calculate info interest rates by subject
            info_cols = ['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']
            info_cols_exist = [c for c in info_cols if c in filtered_df.columns]
            if info_cols_exist:
                top_subjects = filtered_df['subject'].value_counts().head(5).index
                subject_interest_profile = filtered_df[filtered_df['subject'].isin(top_subjects)].groupby('subject')[info_cols_exist].apply(lambda x: (x.notna().mean() * 100).round(1)).to_string()
            else:
                subject_interest_profile = "No info columns available"
            
            data_context = f"""
{filter_context}
TOP 15 SUBJECTS/MAJORS:
{subject_counts}

AVG ATTRACTIVENESS BY SUBJECT:\n{avg_attract_by_subject}

INTEREST PROFILE BY SUBJECT:\n{subject_interest_profile}
"""
            
            prompt = f"""Analyze respondent personas by subject/major:
{data_context}

Provide 3 actionable insights:
1. What are the distinct persona groups and their characteristics?
2. Which subjects have highest/lowest engagement?
3. Targeted engagement strategies for each persona type"""
            st.session_state.insights['personas'] = llm.invoke(prompt).content
        
        # 6. Motivation Drivers
        if 'motivation' in filtered_df.columns and 'attractiveness' in filtered_df.columns:
            motivation_counts = filtered_df['motivation'].value_counts().head(15).to_string()
            motivation_stats_df = filtered_df.groupby('motivation')['attractiveness'].agg(['mean', 'median', 'count', 'std']).round(2).sort_values('mean', ascending=False).head(15)
            motivation_stats = motivation_stats_df.to_string()
            
            # Top motivations with high attractiveness
            high_attract = motivation_stats_df[motivation_stats_df['mean'] >= 7].to_string()
            low_attract = motivation_stats_df[motivation_stats_df['mean'] < 5].to_string()
            
            data_context = f"""
{filter_context}
TOP 15 MOTIVATIONS:\n{motivation_counts}

MOTIVATION STATS (sorted by avg attractiveness):\n{motivation_stats}

High Attractiveness (>=7):\n{high_attract}

Low Attractiveness (<5):\n{low_attract}
"""
            
            prompt = f"""Analyze motivation factors driving attractiveness:
{data_context}

Provide 3 actionable insights:
1. What motivates respondents most to rate highly?
2. Which motivations have lowest attractiveness ratings?
3. Recommendations for leveraging top motivators in employer branding"""
            st.session_state.insights['motivation'] = llm.invoke(prompt).content
        
        # 7. Information Gap Analysis
        if all(col in filtered_df.columns for col in ['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']) and 'motivation' in filtered_df.columns:
            info_cols = ['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']
            
            # Want to learn rates
            info_binary = filtered_df[info_cols].notna().astype(int)
            want_to_learn = (info_binary.sum() / len(info_binary) * 100).round(1)
            
            # Motivation rates
            motivation_mapping = {
                'Compensation & Benefits': ['Competitive salary', 'Compensation and benefits', 'Sign-on bonus', 'Performance bonus'],
                'Career Progression': ['Career development and growth opportunities', 'Career progression', 'Professional development'],
                'Work-Life Balance': ['Work-life balance', 'Company culture', 'Flexible working arrangements', 'Team environment'],
                'Types of Roles': ['Challenging and interesting work', 'Job security', 'Variety of roles', 'Role diversity'],
                'Application Process': ['Recruitment process', 'Interview process']
            }
            
            motivation_rates = (filtered_df['motivation'].value_counts() / len(filtered_df) * 100).round(2)
            
            aggregated_motivation = {}
            for category, keywords in motivation_mapping.items():
                total = sum(motivation_rates[motivation_rates.index.str.contains(k, case=False, na=False)].sum() for k in keywords)
                aggregated_motivation[category] = total
            
            gap_df = pd.DataFrame({
                'Want to Learn (%)': want_to_learn,
                'Motivates to Apply (%)': pd.Series(aggregated_motivation)
            }).round(1)
            gap_df['Gap'] = (gap_df['Want to Learn (%)'] - gap_df['Motivates to Apply (%)']).round(1)
            
            data_context = f"""
{filter_context}
INFORMATION GAP ANALYSIS:
{gap_df.to_string()}

INTERPRETATION:
- Positive Gap: Respondents want to learn but it's NOT a top motivator
- Negative Gap: Topic motivates applications even if not frequently selected as 'want to learn'
"""
            
            prompt = f"""Analyze the information gap between curiosity and motivation:
{data_context}

Provide 3 actionable insights:
1. What information are respondents curious about but doesn't drive applications?
2. What does drive applications despite lower curiosity?
3. Content strategy recommendations to close the gap"""
            st.session_state.insights['gap_analysis'] = llm.invoke(prompt).content
        
        # 8. Maturity Shift
        if 'year' in filtered_df.columns:
            year_order = ['Year 1', 'Year 2', 'Year 3', 'Year 4']
            filtered_years = [y for y in year_order if y in filtered_df['year'].unique()]
            
            year_counts = filtered_df['year'].value_counts().reindex(filtered_years).to_string()
            
            # Interest rates by year
            info_cols = ['info_roles', 'info_career', 'info_comp', 'info_culture', 'info_process']
            info_cols_exist = [c for c in info_cols if c in filtered_df.columns]
            
            if info_cols_exist:
                year_interest = {}
                for year in filtered_years:
                    year_df = filtered_df[filtered_df['year'] == year]
                    year_interest[year] = {col: (year_df[col].notna().mean() * 100).round(1) for col in info_cols_exist}
                year_interest_df = pd.DataFrame(year_interest).T
                interest_by_year = year_interest_df.to_string()
            else:
                interest_by_year = "No info columns available"
            
            # Attractiveness by year
            if 'attractiveness' in filtered_df.columns:
                attract_by_year = filtered_df.groupby('year')['attractiveness'].agg(['mean', 'count']).reindex(filtered_years).round(2).to_string()
            else:
                attract_by_year = "No attractiveness data"
            
            data_context = f"""
{filter_context}
RESPONDENTS BY YEAR:\n{year_counts}

AVERAGE ATTRACTIVENESS BY YEAR:\n{attract_by_year}

INTEREST RATES BY YEAR:\n{interest_by_year}
"""
            
            prompt = f"""Analyze how interests evolve across academic years:
{data_context}

Provide 3 actionable insights:
1. How do information interests change from Year 1 to Year 4?
2. What should be the messaging strategy for each year?
3. Recommendations for year-targeted career outreach"""
            st.session_state.insights['maturity'] = llm.invoke(prompt).content

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
        render_chart_card("📊 Status Distribution", lambda: chart_status_distribution(filtered_df), insight_key="status")
    
    with row1_col2:
        render_chart_card("📊 Drop-off Analysis", lambda: chart_dropoff_analysis(filtered_df), insight_key="dropoff")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2: Question Correlation Heatmap & Partial Prediction Model
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        render_chart_card("📊 Question Correlation Heatmap", lambda: chart_question_correlation_heatmap(filtered_df), insight_key="correlation")
    
    with row2_col2:
        render_chart_card("📊 Factors Driving Partial Responses", lambda: chart_partial_prediction_model(filtered_df), insight_key="partial_drivers")

# Tab 2: Survey Results
with tab_results:
    
    # Section 1: Interest-Based Personas (2 plots side-by-side)
    st.markdown("### 📊 Interest-Based Personas")
    col1, col2 = st.columns(2)
    with col1:
        plot_interest_personas_heatmap(filtered_df)
    with col2:
        plot_interest_personas_distribution(filtered_df)
    
    # Insights for Interest-Based Personas
    if "personas" in st.session_state.insights:
        with st.expander("💡 View Insights", expanded=True):
            st.markdown(st.session_state.insights["personas"])
    else:
        st.info("💡 Click '✨ Generate Insights' in sidebar to see AI-generated insights")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 2: Motivation Drivers (2 plots side-by-side)
    st.markdown("### 📊 Motivation Drivers")
    col1, col2 = st.columns(2)
    with col1:
        plot_motivation_bar(filtered_df)
    with col2:
        plot_motivation_box(filtered_df)
    
    # Insights for Motivation Drivers
    if "motivation" in st.session_state.insights:
        with st.expander("💡 View Insights", expanded=True):
            st.markdown(st.session_state.insights["motivation"])
    else:
        st.info("💡 Click '✨ Generate Insights' in sidebar to see AI-generated insights")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 3: Information Gap Analysis (2 plots side-by-side)
    st.markdown("### 📊 Information Gap Analysis")
    col1, col2 = st.columns(2)
    with col1:
        plot_information_gap_grouped(filtered_df)
    with col2:
        plot_information_gap_lollipop(filtered_df)
    
    # Insights for Information Gap Analysis
    if "gap_analysis" in st.session_state.insights:
        with st.expander("💡 View Insights", expanded=True):
            st.markdown(st.session_state.insights["gap_analysis"])
    else:
        st.info("💡 Click '✨ Generate Insights' in sidebar to see AI-generated insights")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 4: Maturity Shift (2 plots side-by-side)
    st.markdown("### 📊 Maturity Shift (Year 1 → 4)")
    col1, col2 = st.columns(2)
    with col1:
        plot_maturity_line(filtered_df)
    with col2:
        plot_maturity_heatmap(filtered_df)
    
    # Insights for Maturity Shift
    if "maturity" in st.session_state.insights:
        with st.expander("💡 View Insights", expanded=True):
            st.markdown(st.session_state.insights["maturity"])
    else:
        st.info("💡 Click '✨ Generate Insights' in sidebar to see AI-generated insights")

