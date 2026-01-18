import streamlit as st
import pandas as pd
from data_handlers.data_loader import DataLoader
from data_handlers.data_validator import DataValidator

def render_file_uploader():
    """
    Render file upload component with validation
    
    Returns:
        Validated and cleaned dataframe or None
    """
    st.subheader("📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a file (CSV, Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset for automated visualization and AI analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner("Loading data..."):
                loader = DataLoader()
                df = loader.load_file(uploaded_file)
            
            st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
            
            # Validate
            with st.expander("🔍 Data Quality Report", expanded=False):
                validator = DataValidator(df)
                is_valid, errors, warnings = validator.validate()
                
                if errors:
                    st.error("**Errors:**")
                    for error in errors:
                        st.write(f"❌ {error}")
                
                if warnings:
                    st.warning("**Warnings:**")
                    for warning in warnings:
                        st.write(f"⚠️ {warning}")
                
                # Summary stats
                summary = validator.get_summary()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Numeric Columns", summary['numeric_columns'])
                with col2:
                    st.metric("Categorical Columns", summary['categorical_columns'])
                with col3:
                    st.metric("Missing Values", f"{summary['total_missing']:,}")
                with col4:
                    st.metric("Memory", f"{summary['memory_usage_mb']:.2f} MB")
                
                # Option to clean data
                if st.button("🧹 Clean Data Automatically"):
                    df = validator.clean_data()
                    st.success("✅ Data cleaned!")
                    st.rerun()
            
            # Preview
            with st.expander("👀 Data Preview", expanded=True):
                st.dataframe(df.head(20), width='stretch')
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None
    
    return None
