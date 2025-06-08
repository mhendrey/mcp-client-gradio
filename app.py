import asyncio
from fastmcp import Client
import gradio as gr
from gradio.components.multimodal_textbox import MultimodalValue
import json
import logging
from magika import Magika
from pprint import pformat
from typing import Any, Union, Iterable
import yaml

from llm import LLM_Client, MessageProcessor, HTML_PREFIX, HTML_SUFFIX
from mcp_server_handlers import (
    MCPServerRegistry,
    DefaultHandler,
    BlackforestHandler,
    FetchHandler,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


class MCPClientWrapper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.configs = yaml.safe_load(open("config.yaml"))
        self.default_model = self.configs["models"]["default"]
        self.available_models = self.configs["models"]["available"]

        # MCP Server Registry
        self.registry = MCPServerRegistry()
        for server_name, handler in [
            ("blackforest", BlackforestHandler),
            ("fetch", FetchHandler),
        ]:
            self.registry.add_server(
                server_name, self.configs["mcpServers"][server_name], handler
            )

        self.client = Client(self.registry.get_mcp_config())
        self.tools = loop.run_until_complete(self.list_tools())
        self.llm_client = LLM_Client(self.default_model, self.tools)
        self.message_processor = MessageProcessor()
        self.file_typer = Magika()

    async def list_tools(self) -> list[dict]:
        async with self.client:
            print(f"Client connected: {self.client.is_connected()}")
            tools = await self.client.list_tools()
            print(f"Available tools: {[tool.name for tool in tools]}")
            # Convert them to format consumable by ollama
            tools = [{"type": "function", "function": t.model_dump()} for t in tools]

        return tools

    @staticmethod
    def document_to_markdown(filepath: str) -> str:
        pass

    def process_files(self, uploaded_files: list[str]) -> tuple[str, list[str]]:
        message_images = []
        document_text = ""
        file_types = self.file_typer.identify_paths(uploaded_files)
        for f, file_type in zip(uploaded_files, file_types):
            if file_type.output.group == "image":
                message_images.append(f)
            elif file_type.output.group == "document" or file_type.output.is_text:
                pass  # Call markitdown

    def process_message(
        self,
        message: MultimodalValue,
        history: list[Union[gr.ChatMessage, dict[str, Any]]],
        model_id: str = None,
        think: bool = True,
    ):
        if model_id is None:
            model_id = self.default_model

        if model_id != self.llm_client.model_id:
            self.llm_client.set_model_id(model_id)
        think = think if self.llm_client.is_thinking else False

        message_text = message.get("text", "")
        message_files = message.get("files", [])

        # Deal with files.
        # Image files will be passed to LLM (if multimodal)
        # Other files will have text extracted to be added to the message_text
        message_images = []
        for f in message_files:
            file_type = self.file_typer.identify_path(f).output
            if file_type.group == "image":
                message_images.append(f)
            if file_type.is_text or file_type.group == "document":
                # pass to markitdown
                pass

        # Convert Gradio messages to list[ollama.Message]
        messages = self.message_processor.build_message_history(
            message_text, message_images, history
        )

        self.logger.info("\n\nSubmitting the following messages to LLM")
        for m in messages:
            self.logger.info(f"{pformat(m)}")

        tool_calls = []
        try:
            for chunk, tool_calls in self.llm_client.stream_llm_response(
                messages,
                think,
            ):
                yield chunk
        except Exception as e:
            yield f"Error doing initial LLM call: {e}"

        response = chunk
        tool_responses = []
        if tool_calls:
            for tool_name, tool_args in tool_calls:
                handler = self.registry.get_handler(tool_name)
                display_name = handler.get_tool_display_name(tool_name)

                tool_response = gr.ChatMessage(
                    content=f"{display_name}\n{pformat(tool_args)}",
                    role="assistant",
                    metadata={
                        "title": f"Using {display_name} tool",
                        "status": "pending",
                    },
                )
                yield response + tool_responses + [tool_response]

                # Run the tool
                try:
                    result = loop.run_until_complete(
                        self._execute_tool_call(tool_name, tool_args)
                    )
                    tool_response.metadata["status"] = "done"
                    self.logger.info(
                        f"Finished executing {tool_name} with {pformat(tool_args)}"
                    )
                    self.logger.info(f"\nResult is type {type(result)}\n")
                    tool_responses.append(tool_response)

                    # Use handler to process the response
                    handled_response = handler.handle_tool_response(
                        tool_name, tool_args, result
                    )
                    self.logger.info(f"Finished handle response")
                    if handled_response:
                        tool_responses.append(handled_response)

                    yield response + tool_responses
                except Exception as e:
                    error_msg = (
                        f"Error executing tool {tool_name} with {tool_args}: {e}"
                    )
                    self.logger.error(error_msg)
                    tool_responses.append(
                        gr.ChatMessage(
                            content=error_msg,
                            role="assistant",
                            metadata={"title": "Tool Error", "status": "done"},
                        )
                    )
                    yield response + tool_responses

    async def _execute_tool_call(self, tool_name: str, tool_args: dict):
        """Execute a tool call asynchronously"""
        async with self.client:
            result = await self.client.call_tool(tool_name, tool_args)
            return result[0].text  # result is a list of always 1 and text field?


def create_interface():
    """Create and return the Gradio interface"""
    client = MCPClientWrapper()

    interface = gr.ChatInterface(
        client.process_message,
        type="messages",
        multimodal=True,
        additional_inputs=[
            gr.Dropdown(client.available_models, value=client.default_model),
            gr.Checkbox(label="Think Mode", show_label=True, value=True),
        ],
        chatbot=gr.Chatbot(
            type="messages",
            height=575,
            show_copy_button=True,
            show_copy_all_button=True,
            avatar_images=("assets/user.png", "assets/assistant.png"),
        ),
        textbox=gr.MultimodalTextbox(
            placeholder="Enter prompt and/or upload file",
            file_count="multiple",
        ),
        title="MCP Client Chatbot",
        theme="allenai/gradio-theme",
        # theme="ParityError/Interstellar",
        concurrency_limit=16,
    )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.queue(max_size=64).launch(
        server_name="0.0.0.0",
        show_error=True,
        debug=True,
    )
