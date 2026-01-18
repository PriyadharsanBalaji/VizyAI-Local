from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from llm_agents.ollama_client import OllamaClient  # CHANGED
from llm_agents.prompt_templates import PromptTemplates
import json
import re

class AnalysisState(TypedDict):
    """State for the LangGraph workflow"""
    graphs: List[str]
    data_preview: str
    k_value: int
    graph_interpretations: Dict[str, str]
    selected_graphs: List[Dict]
    final_synthesis: str
    error: str

class LangGraphWorkflow:
    """
    LangGraph workflow for multi-stage LLM analysis
    Uses Ollama local vision model
    """
    
    def __init__(self, ollama_client: OllamaClient, skip_stage1: bool = True):  # CHANGED
        self.client = ollama_client
        self.skip_stage1 = skip_stage1
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AnalysisState)
        
        if not self.skip_stage1:
            workflow.add_node("interpret_all_graphs", self.interpret_all_graphs)
            workflow.set_entry_point("interpret_all_graphs")
            workflow.add_edge("interpret_all_graphs", "select_top_k")
        else:
            workflow.set_entry_point("select_top_k")
        
        workflow.add_node("select_top_k", self.select_top_k_graphs_smart)
        workflow.add_node("deep_analysis", self.deep_analysis_selected)
        workflow.add_node("synthesize", self.synthesize_insights)
        
        workflow.add_edge("select_top_k", "deep_analysis")
        workflow.add_edge("deep_analysis", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def interpret_all_graphs(self, state: AnalysisState) -> AnalysisState:
        """Node 1: Interpret all graphs (SKIPPED by default)"""
        interpretations = {}
        
        print(f"Stage 1: Interpreting {len(state['graphs'])} graphs...")
        
        for idx, graph_path in enumerate(state["graphs"]):
            print(f"  [{idx+1}/{len(state['graphs'])}] {graph_path.split('/')[-1]}")
            try:
                response = self.client.generate_content(
                    prompt=PromptTemplates.GRAPH_INTERPRETATION,
                    image_path=graph_path
                )
                interpretations[graph_path] = response
            except Exception as e:
                interpretations[graph_path] = f"Error: {str(e)}"
        
        state["graph_interpretations"] = interpretations
        return state
    
    def select_top_k_graphs_smart(self, state: AnalysisState) -> AnalysisState:
        """Node 2: Smart graph selection"""
        print(f"Stage 2: Selecting top-{state['k_value']} graphs...")
        
        graph_descriptions = []
        for path in state["graphs"]:
            name = path.replace('\\', '/').split('/')[-1].replace('.png', '')
            
            # Auto-describe based on filename
            if 'correlation' in name:
                desc = "Correlation heatmap of numeric variables"
            elif 'pca' in name:
                desc = "PCA dimensionality reduction visualization"
            elif 'tsne' in name:
                desc = "t-SNE clustering visualization"
            elif 'timeseries' in name:
                desc = "Time series trend analysis"
            elif 'dist_' in name:
                var = name.replace('dist_', '').replace('_', ' ')
                desc = f"Distribution of {var}"
            elif 'box_' in name:
                var = name.replace('box_', '').replace('_', ' ')
                desc = f"Box plot of {var}"
            elif 'cat_' in name:
                var = name.replace('cat_', '').replace('_', ' ')
                desc = f"Categorical analysis of {var}"
            elif 'som' in name:
                desc = "Self-Organizing Map"
            elif 'pair' in name:
                desc = "Pairwise relationship matrix"
            else:
                desc = name.replace('_', ' ')
            
            graph_descriptions.append({'path': path, 'name': name, 'description': desc})
        
        # Build selection prompt
        graph_list = "\n".join([
            f"{i+1}. {g['name']}: {g['description']}"
            for i, g in enumerate(graph_descriptions)
        ])
        
        prompt = PromptTemplates.GRAPH_SELECTION.format(
            n_rows=10,
            data_preview=state['data_preview'],
            graph_list=graph_list,
            k=state['k_value']
        )
        
        try:
            response = self.client.generate_content(prompt)
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response_clean = json_match.group()
            else:
                response_clean = response.strip()
            
            selected = json.loads(response_clean)
            
            # Map to paths
            selected_with_paths = []
            for item in selected.get("selected_graphs", [])[:state['k_value']]:
                matching = next((g for g in graph_descriptions 
                               if g['name'] in item.get('graph_name', '')), None)
                if matching:
                    selected_with_paths.append({
                        'graph_name': matching['name'] + '.png',
                        'graph_path': matching['path'],
                        'reason': item.get('reason', 'Selected')
                    })
            
            state["selected_graphs"] = selected_with_paths
            
        except Exception as e:
            print(f"  Selection error: {e}, using priority fallback")
            priority = ['correlation', 'pca', 'timeseries', 'tsne', 'dist']
            scored = [(sum(1 for kw in priority if kw in g['name'].lower()), g) 
                     for g in graph_descriptions]
            scored.sort(reverse=True)
            
            state["selected_graphs"] = [
                {'graph_name': g['name'] + '.png', 'graph_path': g['path'],
                 'reason': 'Auto-selected'}
                for _, g in scored[:state['k_value']]
            ]
        
        print(f"  Selected: {[s['graph_name'] for s in state['selected_graphs']]}")
        return state
    
    def deep_analysis_selected(self, state: AnalysisState) -> AnalysisState:
        """Node 3: Deep analysis of selected graphs"""
        deep_insights = {}
        
        print(f"Stage 3: Deep analysis of {len(state['selected_graphs'])} graphs...")
        
        for idx, selected in enumerate(state["selected_graphs"]):
            graph_path = selected.get('graph_path')
            graph_name = selected.get('graph_name')
            
            print(f"  [{idx+1}/{len(state['selected_graphs'])}] {graph_name}")
            
            if graph_path:
                try:
                    deep_prompt = f"""Perform comprehensive analysis of this visualization.

Provide:
1. **What the graph shows**: Variables, axes, data representation
2. **Key patterns**: Trends, clusters, correlations, anomalies
3. **Statistical observations**: Distributions, outliers, ranges, central tendencies
4. **Implications**: What these patterns mean for the data
5. **Recommendations**: Actionable insights

Be specific and quantitative where visible.
"""
                    response = self.client.generate_content(deep_prompt, image_path=graph_path)
                    deep_insights[graph_name] = response
                except Exception as e:
                    deep_insights[graph_name] = f"Analysis error: {str(e)}"
        
        state["graph_interpretations"] = deep_insights
        return state
    
    def synthesize_insights(self, state: AnalysisState) -> AnalysisState:
        """Node 4: Final synthesis"""
        print(f"Stage 4: Synthesizing insights...")
        
        insights_text = "\n\n".join([
            f"**{sel['graph_name']}**:\n{state['graph_interpretations'].get(sel['graph_name'], 'N/A')}"
            for sel in state["selected_graphs"]
        ])
        
        prompt = PromptTemplates.FINAL_SYNTHESIS.format(
            graph_insights=insights_text,
            data_preview=state["data_preview"]
        )
        
        try:
            synthesis = self.client.generate_content(prompt)
            state["final_synthesis"] = synthesis
        except Exception as e:
            state["final_synthesis"] = f"Synthesis error: {str(e)}"
            state["error"] = str(e)
        
        return state
    
    def run(self, graphs, data_preview, k_value=5):
        """Execute workflow"""
        print(f"\n{'='*60}")
        print(f"Ollama AI Analysis Workflow")
        print(f"Model: {self.client.model_name}")
        print(f"Graphs: {len(graphs)} total, analyzing top {k_value}")
        print(f"{'='*60}\n")
        
        initial_state = {
            "graphs": graphs,
            "data_preview": data_preview,
            "k_value": k_value,
            "graph_interpretations": {},
            "selected_graphs": [],
            "final_synthesis": "",
            "error": ""
        }
        
        result = self.workflow.invoke(initial_state)
        
        print(f"\n{'='*60}")
        print(f"Analysis Complete!")
        print(f"{'='*60}\n")
        
        return result
