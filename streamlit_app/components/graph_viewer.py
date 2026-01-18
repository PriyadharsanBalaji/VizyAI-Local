import streamlit as st
from pathlib import Path
from typing import List

def render_graph_gallery(graph_paths: List[str], 
                        columns: int = 3,
                        show_download: bool = True):
    """
    Render gallery of generated graphs
    
    Args:
        graph_paths: List of paths to graph images
        columns: Number of columns in gallery
        show_download: Show download button for each graph
    """
    st.subheader(f"📊 Generated Visualizations ({len(graph_paths)})")
    
    # Search/filter
    search = st.text_input("🔍 Filter graphs", placeholder="e.g., correlation, distribution")
    
    if search:
        filtered_paths = [
            p for p in graph_paths 
            if search.lower() in Path(p).stem.lower()
        ]
    else:
        filtered_paths = graph_paths
    
    st.write(f"Showing {len(filtered_paths)} graphs")
    
    # Gallery layout
    cols = st.columns(columns)
    
    for idx, graph_path in enumerate(filtered_paths):
        with cols[idx % columns]:
            graph_name = Path(graph_path).stem.replace('_', ' ').title()
            
            st.image(graph_path, caption=graph_name, width='stretch')
            
            if show_download:
                with open(graph_path, 'rb') as f:
                    st.download_button(
                        label=f"⬇️ Download",
                        data=f,
                        file_name=Path(graph_path).name,
                        mime="image/png",
                        key=f"download_{idx}"
                    )

def render_graph_with_insight(graph_path: str, 
                              insight: str,
                              show_graph: bool = True):
    """
    Render single graph with AI insight
    
    Args:
        graph_path: Path to graph image
        insight: AI-generated insight text
        show_graph: Whether to display the graph image
    """
    graph_name = Path(graph_path).stem.replace('_', ' ').title()
    
    with st.container():
        st.markdown(f"### {graph_name}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if show_graph and Path(graph_path).exists():
                st.image(graph_path, width='stretch')
        
        with col2:
            st.markdown("**AI Insight:**")
            st.write(insight)
        
        st.markdown("---")
