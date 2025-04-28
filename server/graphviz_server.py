# graphviz_server.py
from mcp.server.fastmcp import FastMCP, Image, Context
import graphviz
from PIL import Image as PILImage
import sys
import io
mcp = FastMCP("Graphviz")

# In-memory store for graphs
graphs = {}
resource_registry  = {}
@mcp.tool()
def list_resources() -> list:
    return [
        {"uri": uri, "mimeType": entry["mimeType"], "filepath": entry["filepath"]}
        for uri, entry in resource_registry.items()
    ]
def get_resource(uri: str) -> dict:
    entry = resource_registry.get(uri)
    if not entry:
        raise Exception("Resource not found")
    # You might want to return data in the embedded resource MCP format
    return {
        "type": "resource",
        "resource": {
            "uri": uri,
            "mimeType": entry["mimeType"],
            "bytes": entry["bytes"]
        }
    }

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

    graphs[graph_name].node(node_name)
    return f"Node '{node_name}' added to graph '{graph_name}'."

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

    graphs[graph_name].edge(from_node, to_node)
    return f"Edge from '{from_node}' to '{to_node}' added in graph '{graph_name}'."

@mcp.tool()
def update_graph_image(graph_name: str) -> str:
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
    buffer = io.BytesIO()
    output_path = graphs[graph_name].render(format='png', cleanup=True)
    image_bytes = buffer.getvalue()

    resource_uri = f"resource://screenshot/{output_path}"
    resource_registry[resource_uri] = {
        "filepath": output_path,
        "mimeType": "image/jpeg",
        "bytes": image_bytes,
    }
    return f"Done! Check the output at {output_path}"

@mcp.tool()
def display_graph(resource_uri: str) -> dict:
    entry = resource_registry.get(resource_uri)
    if not entry:
        raise Exception("Resource not found")
    return Image(data=entry["bytes"], format="jpeg")


if __name__ == "__main__":
    print("Starting Graphviz server...")
    mcp.run(transport="sse")
