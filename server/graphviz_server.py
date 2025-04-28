# graphviz_server.py
from mcp.server.fastmcp import FastMCP, Image, Context
import graphviz
import sys
import io
import mcp.types as types
from tkfontawesome import icon_to_image
import requests
from tkfontawesome import icon_to_image
mcp = FastMCP("Graphviz")

# In-memory store for graphs
graphs = {}
resource_registry  = {}

@mcp.resource("dir://file")
async def list_resources() -> list[dict]:
    return [
        {"uri": uri, "mimeType": entry["mimeType"], "filepath": entry["filepath"]}
        for uri, entry in resource_registry.items()
    ]

@mcp.tool()
async def get_resource(uri: str) -> dict:
    """
    Retrieves a resource from the resource registry based on its URI.
    This function looks up the provided URI in the resource registry and 
    returns the resource's metadata and content in a structured format. 
    If the resource is not found, an exception is raised.
    Args:
        uri (str): The unique identifier (URI) of the resource to retrieve.
    Returns:
        dict: A dictionary containing the resource's type, URI, MIME type, 
              and content bytes.
    Raises:
        Exception: If the resource with the specified URI is not found.
    """
    if not isinstance(uri, str):
        raise ValueError("The 'uri' parameter must be a string.")

    entry = resource_registry.get(uri)
    if not entry:
        raise Exception("Resource not found")

    return {
        "type": "resource",
        "resource": {
            "uri": uri,
            "mimeType": entry["mimeType"],
            "bytes": entry["bytes"]
        }
    }

@mcp.tool()
async def create_graphviz_graph(graph_name: str) -> str:
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

@mcp.tool()
async def add_node(graph_name: str, node_name: str) -> str:
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
async def list_all_graph_in_memory() -> str:
    """
    Returns a list of all graph names stored in memory.
    
    Returns:
        A string containing the names of all graphs.
    """
    if not graphs:
        return "No graphs available."

    return ", ".join(graphs.keys())

@mcp.tool()
def add_edge(graph_name: str, from_node: str, to_node: str, label: str = None) -> str:
    """
    Adds an edge between two nodes in the specified graph, with an optional label.
    
    Args:
        graph_name: The name of the graph.
        from_node: The starting node of the edge.
        to_node: The ending node of the edge.
        label: The label for the edge (optional).
    
    Returns:
        A confirmation message.
    """
    if graph_name not in graphs:
        return f"Graph '{graph_name}' does not exist."

    if label:
        graphs[graph_name].edge(from_node, to_node, label=label)
    else:
        graphs[graph_name].edge(from_node, to_node)
    
    return f"Edge from '{from_node}' to '{to_node}' added in graph '{graph_name}' with label '{label}'." if label else f"Edge from '{from_node}' to '{to_node}' added in graph '{graph_name}'."

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
def render_graph(graph_name: str) -> str:
    """
    Renders the specified graph to a PNG file and stores its bytes in the resource registry.
    
    Args:
        graph_name: The name of the graph to be rendered.
    
    Returns:
        A confirmation message with the output file path.
    """
    if graph_name not in graphs:
        return f"Graph '{graph_name}' does not exist."

    # Render the graph to a PNG file
    output_path = graphs[graph_name].render(format='png', cleanup=True)

    # Read the rendered PNG file as bytes
    with open(output_path, 'rb') as f:
        image_bytes = f.read()

    # Store the image bytes in the resource registry
    resource_uri = f"file://graph_images/{output_path}"
    resource_registry[resource_uri] = {
        "filepath": output_path,
        "mimeType": "image/png",
        "bytes": image_bytes,
    }

    return f"Graph '{graph_name}' rendered successfully. Check the output at {resource_uri}"

@mcp.tool()
def set_layout(graph_name: str, layout: str = "dot") -> str:
    """
    Sets the layout engine for the specified graph. Defaults to 'dot' if no layout is specified.

    Args:
        graph_name: The name of the graph to update.
        layout: The layout engine to use. Available options are:
            - 'dot': Hierarchical or layered drawings of directed graphs.
            - 'neato': Spring model layouts for undirected graphs.
            - 'fdp': Force-directed layouts for undirected graphs.
            - 'sfdp': Scalable force-directed layouts for large undirected graphs.
            - 'twopi': Radial layouts.
            - 'circo': Circular layouts.
            - 'osage': Clustered graph layouts.
            - 'patchwork': Treemap-style layouts.
            - 'dot_static': Static hierarchical layout.

    Returns:
        A confirmation message.

    Raises:
        ValueError: If the specified layout is not a valid option.
    """
    valid_layouts = {'dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo', 'osage', 'patchwork', 'dot_static'}
    if layout not in valid_layouts:
        raise ValueError(f"Invalid layout '{layout}'. Available layouts are: {', '.join(valid_layouts)}.")

    if graph_name not in graphs:
        return f"Graph '{graph_name}' does not exist."

    graphs[graph_name].engine = layout
    
    return f"Layout for graph '{graph_name}' set to '{layout}'."

@mcp.tool()
async def label_diagram(graph_name: str, title: str, ctx: Context, position: str = "top") -> str:
    """
    Adds a title or label to the specified graph at the given position.

    Args:
        graph_name: The name of the graph to label.
        title: The text of the label or title.
        position: The position of the label. Options are 'top', 'bottom', 'left', 'right'.
                    Defaults to 'top'.

    Returns:
        A confirmation message.
    """
    if graph_name not in graphs:
        return f"Graph '{graph_name}' does not exist."

    valid_positions = {"top", "bottom", "left", "right"}
    if position not in valid_positions:
        return f"Invalid position '{position}'. Valid positions are: {', '.join(valid_positions)}."

    # Add the label based on the position
    if position == "top":
        graphs[graph_name].attr(label=title, labelloc="t")
    elif position == "bottom":
        graphs[graph_name].attr(label=title, labelloc="b")
    elif position == "left":
        graphs[graph_name].attr(label=title, labelloc="l")
    elif position == "right":
        graphs[graph_name].attr(label=title, labelloc="r")

    await ctx.info(f"Label '{title}' added to graph '{graph_name}' at position '{position}'.")
    return f"Label '{title}' added to graph '{graph_name}' at position '{position}'."

@mcp.tool()
def display_graph(resource_uri: str) -> Image:
    """
    Displays a graph image based on the provided resource URI.

    This tool retrieves a graph resource from the resource registry using the given URI 
    and returns it as an image in png format. If the resource is not found, an exception 
    is raised.

    Args:
        resource_uri (str): The URI of the resource to retrieve.

    Returns:
        Image: The graph image in png format.

    Raises:
        Exception: If the resource is not found in the registry.
    """
    entry = resource_registry.get(resource_uri)
    if not entry:
        raise Exception("Resource not found")
    return Image(data=entry["bytes"], format="png")

@mcp.tool()
async def find_icon(graph_name: str, node_name: str, search_query: str, ctx: Context, color: str = "black", size: int = 64) -> str:
    """
    Searches for an icon in the TkFontAwesome library and assigns it to a specific node in the specified graph.

    Args:
        graph_name: The name of the graph containing the node.
        node_name: The name of the node to which the icon will be applied.
        search_query: The search query to find the icon.
        color: The color of the icon. Defaults to "black".
        size: The size of the icon. Defaults to 64.

    Returns:
        A confirmation message.
    """

    if graph_name not in graphs:
        return f"Graph '{graph_name}' does not exist."

    graph = graphs[graph_name]

    # Check if the node exists in the graph
    if node_name not in graph.body:
        return f"Node '{node_name}' does not exist in graph '{graph_name}'."

    try:
        # Generate an icon image using TkFontAwesome
        icon_image = icon_to_image(search_query, size=size, color=color)

        # Save the icon to a temporary file
        buffer = io.BytesIO()
        icon_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Apply the icon to the node
        graph.node(node_name, image=buffer, shape="none", label="")
        await ctx.info(f"Icon applied to node '{node_name}' in graph '{graph_name}' with color '{color}' and size '{size}'.")
        return f"Icon applied to node '{node_name}' in graph '{graph_name}' with color '{color}' and size '{size}'."

    except Exception as e:
        return f"Error while searching for icon: {str(e)}"

if __name__ == "__main__":
    print("Starting Graphviz server...")
    mcp.run(transport="sse")
