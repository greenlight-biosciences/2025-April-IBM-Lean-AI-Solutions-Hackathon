# graphviz_server.py
from mcp.server.fastmcp import FastMCP, Image, Context
import graphviz
from PIL import Image as PILImage
import sys

mcp = FastMCP("Graphviz")

# In-memory store for graphs
graphs = {}

@mcp.tool()
async def create_graphviz_graph(graph_name: str, ctx: Context) -> str:
    """
    Creates a new Graphviz Digraph object and stores it in memory.
    
    Args:
        graph_name: The name/identifier for the graph.
    
    Returns:
        A confirmation message.
    """
    if graph_name in graphs:
        return f"Graph '{graph_name}' already exists."

    print(f"Current Graph {graphs}", file=sys.stderr)
    graphs[graph_name] = graphviz.Digraph(name=graph_name)
    print(f"Graph '{graph_name}' created. {graphs}", file=sys.stderr)

    await ctx.info(f"Graph '{graph_name}' created.")
    return f"Graph '{graph_name}' created successfully."

@mcp.tool()
async def add_node(graph_name: str, node_name: str, ctx: Context) -> str:
    """
    Adds a node to the specified graph.
    
    Args:
        graph_name: The name of the graph to which the node will be added.
        node_name: The name of the node to be added.
    
    Returns:
        A confirmation message.
    """
    if graph_name not in graphs:
        return f"Graph '{graph_name}' does not exist."
    print(f"Added a node {node_name}", file=sys.stderr)
    graphs[graph_name].node(node_name)
    await ctx.info(f"Node '{node_name}' added to graph '{graph_name}'.")
    return f"Node '{node_name}' added to graph '{graph_name}'."

@mcp.tool()
async def list_all_graphs(ctx: Context) -> str:
    """
    Returns a list of all graph names stored in memory.
    
    Returns:
        A string containing the names of all graphs.
    """
    if not graphs:
        return "No graphs available."
    # print(f"{", ".join(graphs.keys())}", file=sys.stderr)
    await ctx.info(", ".join(graphs.keys()))
    return ", ".join(graphs.keys())

@mcp.tool()
def add_edge(graph_name: str, from_node: str, to_node: str, ctx: Context) -> str:
    """
    Adds an edge between two nodes in the specified graph.
    
    Args:
        graph_name: The name of the graph.
        from_node: The starting node of the edge.
        to_node: The ending node of the edge.
    
    Returns:
        A confirmation message.
    """
    if graph_name not in graphs:
        return f"Graph '{graph_name}' does not exist."
    print(f"Added an edge from {from_node} to {to_node}", file=sys.stderr)
    graphs[graph_name].edge(from_node, to_node)
    return f"Edge from '{from_node}' to '{to_node}' added in graph '{graph_name}'."

@mcp.tool()
def render_graph_image(graph_name: str) -> str:
    """
    Renders the specified graph to a PNG file and returns the file path.
    
    Args:
        graph_name: The name of the graph to be rendered.
    
    Returns:
    A confirmation message with the output file path.
    """
    if graph_name not in graphs:
        return f"Graph '{graph_name}' does not exist."

    # Render the PNG to a temporary file
    output_path = graphs[graph_name].render(format='png', cleanup=True)
    print(f"Rendered graph '{graph_name}' to {output_path}", file=sys.stderr)
    return f"Done! Check the output at {output_path}"

# @mcp.tool()
# def display_graph() -> Image:
#     """Load an image from disk"""
#     return Image(path='/home/mihirkestur/2025-April-IBM-Lean-AI-Solutions-Hackathon/test2.gv.png')
# @mcp.tool()
# def create_thumbnail() -> Image:
#     """Create a thumbnail from an image"""
#     img = PILImage.open('/home/mihirkestur/2025-April-IBM-Lean-AI-Solutions-Hackathon/test2.gv.png')
#     print(f"Create thumbnail", file=sys.stderr)
#     img.thumbnail((100, 100))
#     print(f"{img}", file=sys.stderr)
#     return Image(data=img.tobytes(), format="png")

@mcp.tool()
def delete_node(graph_name: str, node_name: str) -> str:
    """Delete a node and its edges from the graph."""
    if graph_name not in graphs:
        return f"Graph '{graph_name}' does not exist."

    original_graph = graphs[graph_name]
    new_graph = type(original_graph)(graph_name)  # preserve Graph or Digraph

    # Copy graph-level attributes
    new_graph.attr(**original_graph.graph_attr)
    new_graph.node_attr.update(original_graph.node_attr)
    new_graph.edge_attr.update(original_graph.edge_attr)

    for line in original_graph.body:
        # Skip any node or edge involving the node to be deleted
        if node_name in line:
            continue
        new_graph.body.append(line)
    print(f"Deleted node {node_name}", file=sys.stderr)
    graphs[graph_name] = new_graph
    return f"Node '{node_name}' and its edges deleted from graph '{graph_name}'."


@mcp.tool()
def delete_edge(graph_name: str, from_node: str, to_node: str) -> str:
    """Delete an edge from the graph."""
    if graph_name not in graphs:
        return f"Graph '{graph_name}' does not exist."

    original_graph = graphs[graph_name]
    new_graph = type(original_graph)(graph_name)

    # Copy graph-level attributes
    new_graph.attr(**original_graph.graph_attr)
    new_graph.node_attr.update(original_graph.node_attr)
    new_graph.edge_attr.update(original_graph.edge_attr)

    # Determine arrow symbol used in the DOT format
    arrow = "->" if isinstance(original_graph, graphviz.Digraph) else "--"
    edge_pattern = f"{from_node} {arrow} {to_node}"

    for line in original_graph.body:
        # Skip the specific edge
        if edge_pattern in line:
            continue
        new_graph.body.append(line)
    print(f"Deleted edge from {from_node} to {to_node}", file=sys.stderr)
    graphs[graph_name] = new_graph
    return f"Edge from '{from_node}' to '{to_node}' deleted from graph '{graph_name}'."

if __name__ == "__main__":
    print("Starting Graphviz server...")
    mcp.run(transport="sse")
