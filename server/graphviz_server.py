# graphviz_server.py
from mcp.server.fastmcp import FastMCP
import graphviz

mcp = FastMCP("Graphviz")

# In-memory store for graphs
graphs = {}

@mcp.tool()
def create_graphviz_graph(graph_name: str) -> str:
    """
    Creates a new Graphviz Digraph object and stores it in memory.
    
    Args:
        graph_name: The name/identifier for the graph.
    
    Returns:
        A confirmation message.
    """
    if graph_name in graphs:
        return f"Graph '{graph_name}' already exists."

    graphs[graph_name] = graphviz.Digraph(name=graph_name)
    return f"Graph '{graph_name}' created successfully."

if __name__ == "__main__":
    mcp.run(transport="sse")
