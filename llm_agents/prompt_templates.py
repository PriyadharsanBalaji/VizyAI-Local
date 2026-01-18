class PromptTemplates:
    
    GRAPH_INTERPRETATION = """
You are a data analyst examining a visualization. Analyze this graph and provide insights.

**Task**: Describe what you observe in this graph in 3-4 concise sentences. Focus on:
1. What the graph shows (variables, relationships)
2. Key patterns, trends, or anomalies
3. Statistical significance if visible

Keep your response factual and specific to what's visible in the graph.
"""

    GRAPH_SELECTION = """
You are selecting the most informative visualizations for data analysis.

**Dataset Preview** (first {n_rows} rows):
{data_preview}

**Available Graphs**:
{graph_list}

**Task**: Rank the top {k} graphs that would provide the MOST valuable insights for this dataset. 
Consider:
- Relevance to the data types present
- Ability to reveal relationships and patterns
- Complementary information (don't pick redundant graphs)

**Output format** (JSON):
{{
  "selected_graphs": [
    {{"graph_name": "graph1.png", "reason": "brief reason"}},
    {{"graph_name": "graph2.png", "reason": "brief reason"}}
  ]
}}

Respond with ONLY valid JSON, no markdown formatting.
"""

    FINAL_SYNTHESIS = """
You are a senior data scientist providing strategic recommendations.

**Context**: Analysis of a dataset with multiple visualizations.

**Individual Graph Insights**:
{graph_insights}

**Dataset Sample**:
{data_preview}

**Task**: Synthesize all insights and provide:

## Key Findings
List 3-5 most important discoveries from the data.

## Patterns & Relationships
Describe cross-graph patterns and correlations observed.

## Actionable Recommendations
Provide specific next steps or decisions based on the analysis.

## Data Quality Notes
Any concerns or limitations observed in the data.

Be specific and actionable. If this is sales/business data, provide business recommendations.
If scientific data, suggest further analyses. Tailor to the data type.
"""
