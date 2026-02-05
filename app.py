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


def chart_time_distribution(filtered_df: pd.DataFrame):
    """Time spent distribution chart."""
    if 'status' not in filtered_df.columns:
        return
    
    status_colors = {'Complete': '#2E7D32', 'Partial': '#FF9800', 'Disqualified': '#D32F2F'}
    
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=100)
    plot_df = filtered_df[filtered_df['duration_sec'] <= 600]
    
    sns.boxenplot(
        x='duration_sec', y='status', data=plot_df,
        order=['Complete', 'Partial', 'Disqualified'],
        palette=status_colors, ax=ax
    )
    ax.set_xlabel("Seconds", fontsize=11)
    ax.set_ylabel("", fontsize=11)
    sns.despine()
    st.pyplot(fig)


def chart_year_qualification(filtered_df: pd.DataFrame):
    """Year vs Qualification heatmap."""
    if 'year' not in filtered_df.columns or 'qualification' not in filtered_df.columns:
        return
    
    consistency_df = pd.crosstab(filtered_df['year'], filtered_df['qualification'])
    if consistency_df.empty:
        return
    
    # Auto-scale based on data dimensions
    n_rows = len(consistency_df)
    n_cols = len(consistency_df.columns)
    fig_width = max(4, min(6, n_cols * 0.6))
    fig_height = max(2.5, min(4, n_rows * 0.5))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    sns.heatmap(
        consistency_df, annot=True, fmt='d', cmap="Blues",
        ax=ax, cbar=False, annot_kws={"size": 8}, linewidths=0.5
    )
    ax.tick_params(axis='x', rotation=0, labelsize=6)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel("", fontsize=10)
    ax.set_ylabel("", fontsize=10)
    st.pyplot(fig)


def chart_attractiveness(filtered_df: pd.DataFrame):
    """Attractiveness score histogram."""
    if 'attractiveness' not in filtered_df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=100)
    sns.histplot(data=filtered_df, x='attractiveness', bins=10, kde=True, color='#2E7D32', ax=ax)
    ax.set_xlim(1, 10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    sns.despine()
    st.pyplot(fig)


def chart_top_majors(filtered_df: pd.DataFrame):
    """Top 8 majors bar chart."""
    if 'subject' not in filtered_df.columns:
        return
    
    top_majors = filtered_df['subject'].value_counts().head(8)
    
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=100)
    sns.barplot(
        y=top_majors.index, x=top_majors.values,
        hue=top_majors.index, palette="viridis", ax=ax, legend=False
    )
    ax.set_xlabel("Count", fontsize=11)
    ax.set_ylabel("", fontsize=11)
    sns.despine()
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
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        render_chart_card("📊 Time Spent Distribution", lambda: chart_time_distribution(filtered_df), "graph1")
    
    with row1_col2:
        render_chart_card("📊 Year vs Qualification", lambda: chart_year_qualification(filtered_df), "graph2")
    
    render_placeholders()

# Tab 2: Survey Results
with tab_results:
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        render_chart_card("📊 Attractiveness Score (1-10)", lambda: chart_attractiveness(filtered_df), "graph3")
    
    with row2_col2:
        render_chart_card("📊 Top 8 Majors", lambda: chart_top_majors(filtered_df), "graph4")
