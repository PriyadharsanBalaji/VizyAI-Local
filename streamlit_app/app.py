import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from data_handlers.data_loader import DataLoader
from visualization.viz_generator import VisualizationGenerator
from llm_agents.ollama_client import OllamaClient  # CHANGED
from llm_agents.langgraph_workflow import LangGraphWorkflow
from utils.cache_manager import CacheManager

st.set_page_config(
    page_title="AI Visualization Insights Tool - Ollama Local",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'graphs' not in st.session_state:
    st.session_state.graphs = []
if 'results' not in st.session_state:
    st.session_state.results = None

def main():
    st.title("📊 AI-Powered Visualization & Insights Tool")
    st.markdown("**Ollama Local Edition** - Upload data, generate visualizations, get AI insights (fully local!)")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ollama status
        st.subheader("🖥️ Ollama Status")
        
        client = OllamaClient()
        status = client.get_rate_limit_status()
        
        if status['status'] == 'local_running':
            st.success(f"✅ Connected: {status['model']}")
        else:
            st.error("❌ Ollama not running")
            st.info("Start Ollama service or check config")
        
        st.metric("Model", Config.OLLAMA_MODEL)
        st.metric("Rate Limits", "None (Local!)")
        
        st.markdown("---")
        
        k_value = st.slider("Top-K Graphs for Deep Analysis", 
                           Config.MIN_K_VALUE, Config.MAX_K_VALUE, 
                           Config.DEFAULT_K_VALUE)
        
        st.markdown("---")
        st.markdown("**💡 Advantages:**")
        st.markdown("✅ No API costs")
        st.markdown("✅ No rate limits")
        st.markdown("✅ Complete privacy")
        st.markdown("✅ Offline capable")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📁 Upload Data", "📈 Visualizations", "🤖 AI Insights", "📋 Final Report"]
    )
    
    with tab1:
        st.header("Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file:
            loader = DataLoader()
            df = loader.load_file(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
            st.dataframe(df.head(10))
            
            if st.button("🚀 Generate Visualizations", type="primary"):
                with st.spinner("Generating visualizations..."):
                    viz_gen = VisualizationGenerator(df)
                    graphs = viz_gen.generate_all()
                    st.session_state.graphs = graphs
                    st.session_state.df = df
                    st.session_state.processed = True
                
                st.success(f"✅ Generated {len(graphs)} visualizations!")
                st.rerun()
    
    with tab2:
        if st.session_state.processed:
            st.header(f"Generated Visualizations ({len(st.session_state.graphs)})")
            
            cols = st.columns(3)
            for idx, graph_path in enumerate(st.session_state.graphs):
                with cols[idx % 3]:
                    st.image(graph_path, caption=Path(graph_path).stem, 
                            width='stretch')
        else:
            st.info("👆 Upload data and generate visualizations first")
    
    with tab3:
        if st.session_state.processed:
            st.header("🤖 AI-Powered Analysis (Ollama Local)")
            
            if status['status'] != 'local_running':
                st.error("⚠️ Ollama not running! Start Ollama service first.")
                st.code("# Windows: Ollama should auto-start\n# Or run: ollama serve", language="bash")
            else:
                if st.button("🔍 Run AI Analysis", type="primary"):
                    with st.spinner("Running local AI analysis... This may take a few minutes."):
                        client = OllamaClient()
                        workflow = LangGraphWorkflow(client)
                        
                        df = st.session_state.df
                        data_preview = df.head(Config.SAMPLE_ROWS_FOR_LLM).to_string()
                        
                        results = workflow.run(
                            graphs=st.session_state.graphs,
                            data_preview=data_preview,
                            k_value=k_value
                        )
                        
                        st.session_state.results = results
                    
                    st.success("✅ Analysis complete!")
                    st.rerun()
                
                # Display results
                if st.session_state.results:
                    results = st.session_state.results
                    
                    st.subheader("📊 Selected Top Graphs")
                    for selected in results['selected_graphs']:
                        with st.expander(f"🔹 {selected['graph_name']}"):
                            st.write(f"**Selection Reason**: {selected['reason']}")
                            
                            graph_path = next(
                                (p for p in results['graphs'] if selected['graph_name'] in p),
                                None
                            )
                            if graph_path:
                                st.image(graph_path)
                                st.write("**AI Interpretation:**")
                                st.write(results['graph_interpretations'].get(selected['graph_name'], 'N/A'))
        else:
            st.info("👆 Generate visualizations first")
    
    with tab4:
        if st.session_state.results:
            st.header("📋 Final Synthesis & Recommendations")
            st.markdown(st.session_state.results['final_synthesis'])
            
            st.download_button(
                label="📥 Download Full Report",
                data=generate_report(st.session_state.results),
                file_name="ai_insights_report_ollama.md",
                mime="text/markdown"
            )
        else:
            st.info("👆 Run AI analysis first")

def generate_report(results):
    """Generate markdown report"""
    report = "# AI Visualization Insights Report (Ollama Local)\n\n"
    report += "## Selected Visualizations\n\n"
    
    for selected in results['selected_graphs']:
        report += f"### {selected['graph_name']}\n"
        report += f"**Reason**: {selected['reason']}\n\n"
        report += f"**Analysis**: {results['graph_interpretations'].get(selected['graph_name'], 'N/A')}\n\n"
    
    report += "## Final Synthesis\n\n"
    report += results['final_synthesis']
    
    return report

if __name__ == "__main__":
    main()
