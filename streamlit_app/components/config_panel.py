import streamlit as st
from config import Config

def render_config_sidebar():
    """
    Render configuration panel in sidebar
    
    Returns:
        Dict with user configuration
    """
    st.sidebar.header("⚙️ Configuration")
    
    # Ollama Configuration
    with st.sidebar.expander("🖥️ Ollama Settings", expanded=True):
        ollama_url = st.text_input(
            "Ollama URL",
            value=Config.OLLAMA_BASE_URL,
            help="URL where Ollama is running"
        )
        
        model = st.selectbox(
            "Model",
            ["llama3.2-vision:11b", "llava:7b", "llava:13b", "llama3.2-vision:90b"],
            index=0,
            help="Vision model for analyzing graphs"
        )
        
        st.info("💡 No API key needed - runs locally!")
    
    # Analysis Settings
    with st.sidebar.expander("📊 Analysis Settings", expanded=True):
        k_value = st.slider(
            "Top-K Graphs for Deep Analysis",
            min_value=Config.MIN_K_VALUE,
            max_value=Config.MAX_K_VALUE,
            value=Config.DEFAULT_K_VALUE,
            help="Number of most informative graphs to analyze deeply"
        )
        
        sample_rows = st.number_input(
            "Sample Rows for LLM",
            min_value=5,
            max_value=50,
            value=Config.SAMPLE_ROWS_FOR_LLM,
            help="Number of data rows to show LLM for context"
        )
    
    # Visualization Settings
    with st.sidebar.expander("🎨 Visualization Settings", expanded=False):
        max_graphs = st.number_input(
            "Max Graphs to Generate",
            min_value=10,
            max_value=50,
            value=Config.MAX_GRAPHS_GENERATE,
            help="Limit total number of visualizations"
        )
        
        dpi = st.selectbox(
            "Figure Quality (DPI)",
            [75, 100, 150, 200],
            index=1,
            help="Higher DPI = better quality but larger files"
        )
    
    # Cache Settings
    with st.sidebar.expander("💾 Cache Settings", expanded=False):
        enable_cache = st.checkbox(
            "Enable Caching",
            value=Config.CACHE_GRAPHS,
            help="Cache graphs and LLM responses"
        )
        
        if enable_cache:
            if st.button("🗑️ Clear Cache"):
                from utils.cache_manager import CacheManager
                cache_mgr = CacheManager()
                cache_mgr.clear_cache()
                st.success("Cache cleared!")
    
    st.sidebar.markdown("---")
    
    # Return configuration
    return {
        "ollama_url": ollama_url,
        "model": model,
        "k_value": k_value,
        "sample_rows": sample_rows,
        "max_graphs": max_graphs,
        "dpi": dpi,
        "enable_cache": enable_cache
    }
