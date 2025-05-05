import sys
import os
import unittest
from unittest.mock import Mock

# Add the server directory to the Python path
test_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.join(test_dir, "server")
sys.path.insert(0, server_dir)

# Existing imports
from graphviz_server import find_icon

# Create a mock context object
mock_ctx = Mock()

class TestFindIconFunction(unittest.TestCase):

    def test_find_icon_valid_query(self):
        """Test find_icon with a valid search query."""
        result = find_icon(graph_name="test_graph", node_name="test_node", search_query="topic", ctx=mock_ctx)
        self.assertIsNotNone(result, "The result should not be None for a valid query.")

    def test_find_icon_invalid_query(self):
        """Test find_icon with an invalid search query."""
        result = find_icon(graph_name="test_graph", node_name="test_node", search_query="nonexistent_icon", ctx=mock_ctx)
        self.assertIsNone(result, "The result should be None for an invalid query.")

    def test_find_icon_missing_parameters(self):
        """Test find_icon with missing parameters."""
        with self.assertRaises(TypeError):
            find_icon(graph_name="test_graph", node_name="test_node")

if __name__ == "__main__":
    unittest.main()