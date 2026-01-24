# 📊 GradSingapore Survey Analytics

**Team: Six-Seven**

An interactive Streamlit dashboard for analyzing student survey data. This tool provides insights into survey completion, respondent demographics, and employer attractiveness through visualizations and an AI-powered chat assistant.

## Technology Stack

- **Frontend:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning / AI:** Scikit-learn, LangChain, OpenAI
- **Plotting:** Matplotlib, Seaborn

## Getting Started

### 1. Setup Environment

Create a virtual environment, activate it, and install the required packages.

```bash
# Create and activate the virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # MacOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

This application requires the `sds_datathon_gradsingapore.xlsx` data file.

1.  Create a folder named `Category B Dataset` in the root of the project.
2.  Place the `sds_datathon_gradsingapore.xlsx` file inside this folder.

The final structure should look like this:

```
.
├── Category B Dataset/
│   └── sds_datathon_gradsingapore.xlsx
└── app.py
```

### 3. Configure Secrets

Create a file at `.streamlit/secrets.toml` and add your API credentials.

```toml
# .streamlit/secrets.toml
MODEL_NAME = "glm-4.7"
OPENAI_API_KEY = "your-api-key"
OPENAI_API_BASE = "https://api.z.ai/api/paas/v4/"
```

## Usage

Launch the Streamlit application with the following command:

```bash
streamlit run app.py
```

The application will open in your web browser.
