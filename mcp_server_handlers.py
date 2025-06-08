from abc import ABC, abstractmethod
import gradio as gr
import json
import logging
from typing import Any, Optional


class MCPServerHandler(ABC):
    """Abstract base class for MCP server handlers"""

    def __init__(self, server_name: str, config: dict):
        self.server_name = server_name
        self.config = config
        self.logger = logging.getLogger("__name__")

    @abstractmethod
    def handle_tool_response(
        self,
        tool_name: str,
        tool_args: dict,
        result: str,
    ) -> Optional[gr.ChatMessage]:
        """Handle the response from a tool call. Return `None` if no special handling
        is needed.

        Parameters
        ----------
        tool_name : str
            Name of the tool
        tool_args : dict
            Arguments used during tool invocation
        result : str
            Resulting output of the tool

        Returns
        -------
        Optional[gr.ChatMessage]
            Return None if no special handling is needed
        """
        pass

    def get_tool_display_name(self, tool_name: str) -> str:
        """Get display name for tool

        Parameters
        ----------
        tool_name : str

        Returns
        -------
        str
        """
        return tool_name.replace(f"{self.server_name}_", "").replace("_", " ").title()


class BlackforestHandler(MCPServerHandler):
    """Handler for Blackforest image generation server"""

    def handle_tool_response(
        self, tool_name: str, tool_args: dict, result_str: str
    ) -> Optional[gr.ChatMessage]:
        if tool_name == "blackforest_generate_image":
            try:
                result = json.loads(result_str)
                if result["type"] == "image":
                    return gr.ChatMessage(
                        content=gr.Image(
                            result["tmp_file"],
                            label=result["message"],
                            show_label=True,
                        ),
                        role="assistant",
                    )
                else:
                    return gr.ChatMessage(
                        content=result["message"],
                        role="assistant",
                    )
            except Exception as e:
                self.logger.error(f"Error parsing blackforest response: {e}")
                return gr.ChatMessage(
                    content=f"Error processing image generation: \n{result}\n\n{e}",
                    role="assistant",
                )
        return None


class FetchHandler(MCPServerHandler):
    """Handler for fetch/web scraping server"""

    def handle_tool_response(
        self, tool_name: str, tool_args: dict, result: str
    ) -> Optional[gr.ChatMessage]:
        if tool_name == "fetch_fetch":
            return gr.ChatMessage(
                content=f"Fetched content:\n\n{result}\n\n",
                role="assistant",
            )


class DefaultHandler(MCPServerHandler):
    """Default handler for servers without specific handling.
    Server result should be a `str` in this case."""

    def handle_tool_response(
        self, tool_name: str, tool_args: dict, result: str
    ) -> Optional[gr.ChatMessage]:
        return gr.ChatMessage(
            content=result,
            role="assistant",
        )


class MCPServerRegistry:
    """Registry for MCP servers and their handlers"""

    def __init__(self):
        self.handlers: dict[str, MCPServerHandler] = {}
        self.configs: dict[str, dict] = {}

    def register_handler(self, server_name: str, handler: MCPServerHandler):
        """Register a handler for a specific server"""
        self.handlers[server_name] = handler

    def get_handler(self, tool_name: str) -> MCPServerHandler:
        """Get the appropriate handler for a tool"""
        # Extract server name from tool name (assumes format: servername_toolname)
        server_name = tool_name.split("_")[0] if "_" in tool_name else "default"

        if server_name in self.handlers:
            return self.handlers[server_name]
        else:
            if "default" not in self.handlers:
                self.handlers["default"] = DefaultHandler("default", {})
            return self.handlers["default"]

    def get_mcp_config(self) -> dict:
        """Get the MCP configuration for all registered servers"""
        return {"mcpServers": self.configs}

    def add_server(
        self, server_name: str, config: dict, handler_class: MCPServerHandler = None
    ):
        """Add a new server with optional custom handler"""
        self.configs[server_name] = config
        if handler_class:
            self.register_handler(server_name, handler_class(server_name, config))
        else:
            self.register_handler(server_name, DefaultHandler(server_name, config))
