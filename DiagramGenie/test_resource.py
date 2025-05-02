import io
import time
import asyncio
from PIL import Image
import streamlit as st
from fastmcp import Client
import json

async def get_workflow_image():
    """Fetch a workflow image from an MCP resource with retries and logging"""

    async with Client("http://localhost:8000/sse") as client:
        # Make sure this is a valid URI to a single image file!
        resource_uri = "file://graph_images"
        response = await client.read_resource(resource_uri)
        for content in response:
            img_bytes = json.loads(content.text)[0]['bytes']
            return img_bytes

def get_image():
    asyncio.run(get_workflow_image())