# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
async def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
async def mihir_operation(a: int, b: int) -> int:
    """Custom operation by Mihir"""
    return a * b + 25

if __name__ == "__main__":
    mcp.run(transport="stdio")