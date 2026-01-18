import streamlit as st

def apply_custom_styles():
    """
    Apply custom CSS styling to Streamlit app
    """
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Headers */
        h1 {
            color: #1f77b4;
            font-weight: 700;
        }
        
        h2 {
            color: #2ca02c;
            font-weight: 600;
            margin-top: 2rem;
        }
        
        h3 {
            color: #ff7f0e;
            font-weight: 500;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: 500;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed #1f77b4;
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Dataframe */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #f0f2f6;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        
        /* Success message */
        .element-container div[data-testid="stMarkdownContainer"] div[data-baseweb="notification"] {
            border-radius: 8px;
        }
        
        /* Cards */
        .css-1r6slb0 {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
