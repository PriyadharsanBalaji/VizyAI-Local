import streamlit as st
from typing import Dict, List

def render_insights_panel(results: Dict):
    """
    Render LLM insights in organized panels
    
    Args:
        results: Results from LangGraph workflow
    """
    st.header("🤖 AI-Generated Insights")
    
    # Stage 1: All graph interpretations (if available)
    if results.get('graph_interpretations') and len(results['graph_interpretations']) > len(results.get('selected_graphs', [])):
        with st.expander("📊 Stage 1: Initial Graph Interpretations", expanded=False):
            st.markdown("*Quick interpretation of all generated graphs*")
            
            for graph_path, interpretation in results.get('graph_interpretations', {}).items():
                graph_name = graph_path.split('/')[-1]
                st.markdown(f"**{graph_name}**")
                st.write(interpretation)
                st.markdown("---")
    
    # Stage 2: Selected graphs
    with st.expander("🎯 Stage 2: Top-K Graph Selection", expanded=True):
        st.markdown("*AI selected the most informative graphs*")
        
        for selected in results.get('selected_graphs', []):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{selected['graph_name']}**")
                st.write(f"*Reason:* {selected['reason']}")
            with col2:
                st.metric("Rank", results['selected_graphs'].index(selected) + 1)
    
    # Stage 3: Deep analysis
    with st.expander("🔬 Stage 3: Deep Analysis of Selected Graphs", expanded=True):
        st.markdown("*In-depth analysis of top graphs*")
        
        for selected in results.get('selected_graphs', []):
            graph_name = selected['graph_name']
            deep_insight = results['graph_interpretations'].get(graph_name, 'N/A')
            
            st.markdown(f"### {graph_name}")
            st.write(deep_insight)
            st.markdown("---")
    
    # Stage 4: Final synthesis
    with st.expander("🎓 Stage 4: Final Synthesis & Recommendations", expanded=True):
        st.markdown(results.get('final_synthesis', 'No synthesis available'))

def render_rate_limit_warning(status: Dict):
    """
    Display rate limit warnings (not needed for local Ollama!)
    """
    # Local model - no rate limits!
    st.success("🎉 Running locally - No rate limits!")
