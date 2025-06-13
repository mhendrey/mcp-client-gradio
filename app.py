import asyncio
from dataclasses import dataclass
from docling.document_converter import DocumentConverter
from fastmcp import Client
import gradio as gr
from gradio.components.multimodal_textbox import MultimodalValue
import logging
from magika import Magika
from ollama import Message
from pprint import pformat
from typing import Any, Union
import yaml

from llm import LLM_Client, MessageProcessor
from mcp_server_handlers import (
    MCPServerRegistry,
    DefaultHandler,
    BlackforestHandler,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


@dataclass
class ProcessedFiles:
    images: list[str]
    documents: list[str]


class FileProcessor:
    def __init__(self):
        self.file_typer = Magika()
        self.converter = DocumentConverter()

    def process_files(self, files: list[str]) -> ProcessedFiles:
        """Process uploaded files and categorize them.

        Parameters
        ----------
        files : list[str]
            Paths to uploaded documents

        Returns
        -------
        ProcessedFiles
        """
        images = []
        documents = []

        file_types = self.file_typer.identify_paths(files)
        for filepath, file_type in zip(files, file_types):
            if file_type.output.group == "image":
                images.append(filepath)
            elif file_type.output.is_text:
                documents.append(open(filepath).read())
            elif file_type.output.group == "document":
                document_content = self._extract_document_text(filepath)
                documents.append(document_content)

        return ProcessedFiles(images=images, documents=documents)

    def _extract_document_text(self, filepath: str) -> str:
        try:
            doc = self.converter.convert(filepath).document
            doc_content = doc.export_to_markdown()
        except Exception as e:
            raise Exception(
                f"Error with docling conversion to markdown on {filepath}: {e}"
            )
        return doc_content


class MCPClientWrapper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.configs = yaml.safe_load(open("config.yaml"))
        self.default_model = self.configs["models"]["default"]
        self.available_models = [
            key for key in self.configs["models"] if key != "default"
        ]

        self.fileprocessor = FileProcessor()

        # MCP Server Registry
        self.registry = MCPServerRegistry()
        for server_name, handler in [
            ("blackforest", BlackforestHandler),
            ("fetch", DefaultHandler),
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

    def process_message(
        self,
        message: MultimodalValue,
        history: list[Union[gr.ChatMessage, dict[str, Any]]],
        model_id: str = None,
        think: bool = True,
        system_prompt: str = "",
    ):
        if model_id is None:
            model_id = self.default_model

        if model_id != self.llm_client.model_id:
            self.llm_client.set_model_id(model_id)
        think = think if self.llm_client.models_config[model_id]["thinking"] else False

        message_text = message.get("text", "")
        message_files = message.get("files", [])
        self.logger.info(f"message_files has type {type(message_files)}")

        # Deal with files.
        # Image files will be passed to LLM (if multimodal)
        # Other files will have text extracted to be added to the message_text
        uploaded_files = self.fileprocessor.process_files(message_files)
        for doc in uploaded_files.documents:
            message_text += f"\n\nProvided Document:\n\n{doc}"

        # Convert Gradio messages to list[ollama.Message]
        messages = self.message_processor.build_message_history(
            message_text, uploaded_files.images, history, system_prompt
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

        thinking_tool_responses = []
        thinking_chat = None
        match len(chunk):
            case 1:
                if think:
                    thinking_tool_responses.append(chunk[0])
                else:
                    content_chat = chunk[0]
            case 2:
                thinking_chat, content_chat = chunk
                thinking_tool_responses.append(thinking_chat)

        if tool_calls:
            for tool_name, tool_args in tool_calls:
                handler = self.registry.get_handler(tool_name)
                display_name = handler.get_tool_display_name(tool_name)

                # Hand jamming because the default value of 5000 is way too small
                if tool_name == "fetch_fetch" and "max_length" not in tool_args:
                    tool_args["max_length"] = 999999  # Maximum value it can be

                tool_response = gr.ChatMessage(
                    content=f"{display_name}\n{pformat(tool_args)}",
                    role="assistant",
                    metadata={
                        "title": f"Using {display_name} tool",
                        "status": "pending",
                    },
                )
                yield thinking_tool_responses + [tool_response]

                # Run the tool
                try:
                    result = loop.run_until_complete(
                        self._execute_tool_call(tool_name, tool_args)
                    )
                    tool_response.metadata["status"] = "done"
                    self.logger.info(
                        f"Finished executing {tool_name} with {pformat(tool_args)}"
                    )
                    thinking_tool_responses.append(tool_response)

                    if tool_name == "fetch_fetch":
                        # Append fetched text to end of user's initial request
                        messages[-1] = Message(
                            role="user", content=f"{message_text}\n\n{result}"
                        )
                        try:
                            for fetch_chunk, _ in self.llm_client.stream_llm_response(
                                messages,
                                think,
                            ):
                                yield thinking_tool_responses + fetch_chunk
                        except Exception as e:
                            yield f"Error doing initial LLM call: {e}"

                        match len(fetch_chunk):
                            case 1:
                                if think:
                                    thinking_tool_responses.append(fetch_chunk[0])
                                else:
                                    content_chat = chunk[0]
                            case 2:
                                thinking_chat, content_chat = fetch_chunk
                                thinking_tool_responses.append(thinking_chat)
                    else:
                        handled_response = handler.handle_tool_response(
                            tool_name, tool_args, result
                        )
                        self.logger.info(f"Finished handle response")
                        if handled_response:
                            content_chat = handled_response

                except Exception as e:
                    error_msg = (
                        f"Error executing tool {tool_name} with {tool_args}: {e}"
                    )
                    self.logger.error(error_msg)
                    thinking_tool_responses.append(
                        gr.ChatMessage(
                            content=error_msg,
                            role="assistant",
                            metadata={"title": "Tool Error", "status": "done"},
                        )
                    )
                yield thinking_tool_responses + [content_chat]

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
            gr.Textbox(
                label="System Prompt",
                show_label=True,
                placeholder="Instructs the LLM on how to respond, setting its tone, "
                "role, and guidelines for interactions.",
            ),
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
        # theme="allenai/gradio-theme",
        theme="earneleh/paris",
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
