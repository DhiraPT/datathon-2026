import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
import os
import uuid


# ---------------------------------------------------------
# PAGE CONFIGURATION & SESSION STATE
# ---------------------------------------------------------
st.set_page_config(page_title="GradSingapore Analytics", layout="wide", page_icon="📊")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I can help you analyze the survey data. Ask me about completion rates, specific schools, or attractiveness scores."}]

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "insights" not in st.session_state:
    st.session_state.insights = {}

# ---------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
@st.cache_data
def load_data():
    file_path = "Category B Dataset/sds_datathon_gradsingapore.xlsx"
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return None

    # Standardize Columns
    rename_dict = {
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
    df = df.rename(columns=rename_dict)

    # Feature Engineering
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['submit_time'] = pd.to_datetime(df['submit_time'])
    df['duration_sec'] = (df['submit_time'] - df['start_time']).dt.total_seconds()
    df['is_mobile'] = df['user_agent'].str.contains('Mobile|Android|iPhone', case=False, na=False).astype(int)

    return df

df = load_data()

if df is None:
    st.error("Data file not found. Please ensure the Excel file is in the directory.")
    st.stop()

# ---------------------------------------------------------
# FILTER STATE INITIALIZATION
# ---------------------------------------------------------
all_schools = sorted(df['school'].dropna().unique()) if 'school' in df.columns else []
all_years = sorted(df['year'].dropna().unique()) if 'year' in df.columns else []

if 'school_selector' not in st.session_state:
    st.session_state.school_selector = all_schools

if 'year_selector' not in st.session_state:
    st.session_state.year_selector = all_years

def close_chat():
    st.session_state.chat_open = False

def on_filter_change():
    st.session_state.chat_open = False
    st.session_state.insights = {}

def reset_schools():
    st.session_state.school_selector = all_schools
    on_filter_change()

def reset_years():
    st.session_state.year_selector = all_years
    on_filter_change()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("🔍 Filters")

if st.sidebar.button("✨ Generate Insights"):
    gen_insights = True
else:
    gen_insights = False

# --- FILTER 1: school ---
st.sidebar.subheader("Higher Education School")

# Multiselect Widget
selected_schools = st.sidebar.multiselect(
    "Select schools:",
    options=all_schools,
    key='school_selector',
    on_change=on_filter_change
)
st.sidebar.caption(f"{len(selected_schools)} of {len(all_schools)} selected")

# Reset Button
st.sidebar.button("Select All schools", on_click=reset_schools)

st.sidebar.divider()

# --- FILTER 2: YEAR ---
st.sidebar.subheader("Year of Study")

# Multiselect Widget
selected_years = st.sidebar.multiselect(
    "Select years:",
    options=all_years,
    key='year_selector',
    on_change=on_filter_change
)
st.sidebar.caption(f"{len(selected_years)} of {len(all_years)} selected")

# Reset Button
st.sidebar.button("Select All Years", on_click=reset_years)

# --- APPLY FILTERS ---
mask = pd.Series([True] * len(df))

if 'school' in df.columns:
    mask &= df['school'].isin(selected_schools)
if 'year' in df.columns:
    mask &= df['year'].isin(selected_years)

filtered_df = df[mask]

# Sidebar Metrics
st.sidebar.divider()
st.sidebar.metric("Filtered Responses", f"{len(filtered_df)}")
st.sidebar.metric("Total Responses", f"{len(df)}")

# ---------------------------------------------------------
# AI LOGIC
# ---------------------------------------------------------
def build_analyst_agent(df):
    llm = ChatOpenAI(
        temperature=0,
        model=st.secrets.get("MODEL_NAME", "glm-4.7"),
        openai_api_key=st.secrets.get("OPENAI_API_KEY", "your-api-key"),
        openai_api_base=st.secrets.get("OPENAI_API_BASE", "https://api.z.ai/api/paas/v4/"),
    )

    SYSTEM_PROMPT = """
You are a data analyst.

You are given a pandas DataFrame called `df`.

You MUST output valid Python code only.

Rules:
- Do NOT include explanations outside code
- Do NOT use markdown
- Do NOT print anything
- You may inspect df using df.head() or df.describe()
- If you generate a plot:
    - call plt.clf()
    - save it to 'temp_plot.png'
    - do NOT call plt.show()
- Store your final explanation for the user in:
    final_answer = "<your explanation here>"
"""

    return llm, SYSTEM_PROMPT

def run_analyst_agent(user_question, df):
    llm, system_prompt = build_analyst_agent(df)

    prompt = f"""
{system_prompt}

User question:
{user_question}
"""

    try:
        response = llm.invoke(prompt)
        code = response.content

        # Clean up code block markers if present
        if code.startswith("```python"):
            code = code.replace("```python", "").replace("```", "")
        elif code.startswith("```"):
            code = code.replace("```", "")
        code = code.strip()

        local_vars = {
            "df": df.copy(),
            "pd": pd,
            "plt": plt,
            "sns": sns,
        }

        exec(code, {}, local_vars)

        final_answer = local_vars.get("final_answer", "Analysis completed.")

        # Check if a plot was generated and rename it to keep history
        image_path = None
        if os.path.exists("temp_plot.png"):
            unique_filename = f"chart_{uuid.uuid4().hex}.png"
            os.rename("temp_plot.png", unique_filename)
            image_path = unique_filename

        return final_answer, code, image_path
    except Exception as e:
        return f"Error: {str(e)}", code if 'code' in locals() else "", None

if gen_insights:
    if filtered_df.empty:
        st.error("No data available to generate insights.")
    else:
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

# ---------------------------------------------------------
# CHAT MODAL
# ---------------------------------------------------------
@st.dialog("🤖 AI Assistant")
def show_chat_modal():
    # Create a container for chat history to ensure messages appear above the input
    chat_container = st.container()

    # Display chat messages from history on app rerun
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
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message and spinner inside the chat container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response
            with st.spinner("Analyzing data..."):
                response, code, image_path = run_analyst_agent(prompt, filtered_df)

            # Add assistant response to chat history
            message_data = {"role": "assistant", "content": response, "code": code}
            if image_path:
                message_data["image"] = image_path
            st.session_state.messages.append(message_data)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.write(response)
                if image_path:
                    with open(image_path, "rb") as f:
                        st.image(f.read())

                if code:
                    with st.expander("🛠️ View Generated Code"):
                        st.code(code, language="python")
        st.rerun()

# Floating Button CSS & Logic
if st.session_state.get("chat_open", False):
    show_chat_modal()

# The trigger button (invisible layout hack to position it, or standard button)
# Since pure CSS buttons can't trigger Python easily without components, 
# we place a standard button in a bottom container and use CSS to float it.
with st.container():
    # We assign a specific key or label to target it with CSS if needed, 
    # but for simplicity in Streamlit, we put it in a sidebar or bottom column.
    # To make it truly float and work, we use a regular button and styling.
    col1, col2 = st.columns([0.9, 0.1]) 
    with col2:
        # Use type="primary" to uniquely identify this button for CSS targeting
        if st.button("💬", key="chat_btn", help="Open AI Assistant", type="primary"):
            st.session_state.chat_open = True
            st.rerun()

# Inject styling to float the specific button we just created
st.markdown("""
<style>
/* 
This CSS targets the container of a "primary" button and styles it as a 
Floating Action Button (FAB). This is a more robust method than targeting
all secondary buttons, which was causing the styling to accidentally apply 
to other buttons in the app.
*/
div.element-container:has(button[kind="primary"]) {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
}
div.element-container button[kind="primary"] {
    border-radius: 50%;
    width: 60px;
    height: 60px;
    background-color: #2E7D32;
    color: white;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    font-size: 24px;
    border: none; /* Overriding default primary button border */
}
div.element-container button[kind="primary"]:hover {
    background-color: #1B5E20; /* Darker green on hover */
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# MAIN DASHBOARD CONTENT
# ---------------------------------------------------------
st.title("📊 GradSingapore Survey Analytics")

# Create Tabs
tab_quality, tab_results = st.tabs(["🛠️ Survey Quality", "📈 Survey Results"])

status_colors = {'Complete': '#2E7D32', 'Partial': '#FF9800', 'Disqualified': '#D32F2F'}

# --- TAB 1: SURVEY QUALITY ---
with tab_quality:
    st.subheader("Operational Metrics")

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    if 'status' in filtered_df.columns:
        comp_rate = (filtered_df['status'] == 'Complete').mean() * 100
        disq_count = len(filtered_df[filtered_df['status'] == 'Disqualified'])
    else:
        comp_rate, disq_count = 0, 0
    avg_dur = filtered_df['duration_sec'].median()

    m1.metric("Completion Rate", f"{comp_rate:.1f}%")
    m2.metric("Median Duration", f"{avg_dur:.0f}s")
    m3.metric("Disqualified", f"{disq_count}")

    st.divider()

    # Graphs Row (2 Columns)
    col1, col2 = st.columns(2)

    # Graph 1: Boxen Plot (Time vs Status)
    with col1:
        st.markdown("#### Time Spent Distribution")
        if 'status' in filtered_df.columns:
            fig_boxen, ax_boxen = plt.subplots(figsize=(6, 5))
            plot_df = filtered_df[filtered_df['duration_sec'] <= 600] # Clip outliers
            sns.boxenplot(
                x='duration_sec', y='status', data=plot_df,
                order=['Complete', 'Partial', 'Disqualified'],
                palette=status_colors, ax=ax_boxen
            )
            ax_boxen.set_xlabel("Seconds")
            sns.despine()
            st.pyplot(fig_boxen)
            if 'graph1' in st.session_state.insights:
                st.info(st.session_state.insights['graph1'])

    # Graph 2: Heatmap
    with col2:
        st.markdown("#### Logic Check: Year vs Qualification")
        if 'year' in filtered_df.columns and 'qualification' in filtered_df.columns and not filtered_df.empty:
            consistency_df = pd.crosstab(filtered_df['year'], filtered_df['qualification'])
            fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
            sns.heatmap(consistency_df, annot=True, fmt='d', cmap="Blues", ax=ax_heat, cbar=False)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_heat)
            if 'graph2' in st.session_state.insights:
                st.info(st.session_state.insights['graph2'])

# --- TAB 2: SURVEY RESULTS ---
with tab_results:
    st.subheader("Insights")

    # Graphs Row (2 Columns)
    col3, col4 = st.columns(2)

    # Graph 3: Attractiveness Histogram
    with col3:
        st.markdown("#### Attractiveness Score (1-10)")
        if 'attractiveness' in filtered_df.columns:
            fig_hist, ax_hist = plt.subplots(figsize=(6, 5))
            sns.histplot(data=filtered_df, x='attractiveness', bins=10, kde=True, color='#2E7D32', ax=ax_hist)
            ax_hist.set_xlim(1, 10)
            st.pyplot(fig_hist)
            if 'graph3' in st.session_state.insights:
                st.info(st.session_state.insights['graph3'])

    # Graph 4: Top Majors Bar Chart
    with col4:
        st.markdown("#### Top 10 Majors")
        if 'subject' in filtered_df.columns:
            top_majors = filtered_df['subject'].value_counts().head(10)
            fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
            sns.barplot(y=top_majors.index, x=top_majors.values, hue=top_majors.index, palette="viridis", ax=ax_bar, legend=False)
            ax_bar.set_xlabel("Count")
            st.pyplot(fig_bar)
            if 'graph4' in st.session_state.insights:
                st.info(st.session_state.insights['graph4'])
