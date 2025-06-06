import base64
import gradio as gr
import logging
from magika import Magika
from ollama import chat, Message
from typing import Any, Generator, Union

# Constants
## Thinking Models
THINKING_MODELS = ["qwen3:30b"]
MULTIMODAL_MODELS = ["gemma3:27b-it-qat"]
TOOL_CALLING_MODELS = ["qwen3:30b"]

## Display Thinking output
HTML_PREFIX = "<!DOCTYPE html>\n<html>\n<body>\n<blockquote>"
HTML_SUFFIX = "\n</blockquote>\n</body>\n</html>"

## Recommended values for "/think" mode in Qwen3
QWEN3_THINK_TEMPERATURE = 0.6
QWEN3_THINK_TOP_P = 0.95
QWEN3_THINK_TOP_K = 20
QWEN3_THINK_MIN_P = 0.0
## RECOMMENDED values for "/no_think"
QWEN3_TEMPERATURE = 0.7
QWEN3_TOP_P = 0.8
QWEN3_TOP_K = 20
QWEN3_MIN_P = 0.0

## RECOMMENDED values for Gemma3
GEMMA3_TOP_K = 64
GEMMA3_TOP_P = 0.95


def b64encode_file(filepath: str) -> str:
    return base64.encode(open(filepath, "rb").read()).decode("utf-8")


class MessageProcessor:
    """Handles message processing and formatting"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.magika = Magika()

    @staticmethod
    def extract_message_content(
        msg: Union[gr.ChatMessage, dict[str, Any]],
    ) -> tuple[str, str]:
        """Extract role and content from message. Input is from Gradio ChatInterface"""
        if isinstance(msg, gr.ChatMessage):
            role, content = msg.role, msg.content
        elif isinstance(msg, dict):
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message missing required keys: {msg.keys()}")
            role, content = msg["role"], msg["content"]
        else:
            raise ValueError(f"Invalid message type: {type(msg)}")

        if role not in ["user", "assistant", "system", "developer"]:
            raise ValueError(f"{role=:} is not recognized")

        return role, content

    def build_message_history(
        self,
        current_message: str,
        current_images: list,
        history: list[Union[gr.ChatMessage, dict[str, Any]]],
    ) -> list[Message]:
        """Build messages array to give to Ollama LLM."""
        messages = []
        message = Message(role="dummy")
        for msg in history:
            try:
                # Skip over thinking or tool calling messages
                if msg["metadata"]:
                    continue
                message.role, content = self.extract_message_content(msg)
                # Skip over content that is a Gradio component
                if isinstance(content, gr.Component):
                    continue
                # Add image files to history, but skip over everything else
                elif isinstance(content, tuple):
                    file_type = self.magika.identify_path(content[0])
                    if file_type.output.group == "image":
                        if not message.images:
                            message.images = [b64encode_file(content[0])]
                        else:
                            message.images.append(b64encode_file(content[0]))
                    else:
                        continue
                # This assumes that files are in `history` before associated text content
                elif isinstance(content, str):
                    message.content = content
                    messages.append(message)
                    message = Message(role="dummy")
                else:
                    raise ValueError(
                        f"content field has unexpected type {type(content)}"
                    )
            except ValueError as e:
                self.logger.warning(f"Skipping invalid message {msg}: {e}")
                message = Message(role="dummy")

        user_message = Message(role="user", content=current_message)
        if current_images:
            user_message.images = [b64encode_file(f) for f in current_images]
        messages.append(user_message)

        return messages


class LLM_Client:
    """Handles calls to the LLM"""

    def __init__(
        self,
        model_id,
        tools: list = [],
    ):
        self.model_id = self.set_model_id(model_id)
        self.tools = tools
        self.logger = logging.getLogger(__name__)

    def set_model_id(self, model_id):
        self.model_id = model_id
        self.is_thinking = True if model_id in THINKING_MODELS else False
        self.is_multimodal = True if model_id in MULTIMODAL_MODELS else False
        self.is_tool_calling = True if model_id in TOOL_CALLING_MODELS else False

    def _get_sample_params(self, think: bool = False) -> dict[str, Any]:
        """Get sampling params based upon model and if using think"""
        if self.model_id.startswith("gemma3"):
            return {
                "top_k": GEMMA3_TOP_K,
                "top_p": GEMMA3_TOP_P,
            }
        elif self.model_id.startswith("qwen3"):
            if think:
                return {
                    "temperature": QWEN3_THINK_TEMPERATURE,
                    "top_p": QWEN3_THINK_TOP_P,
                    "top_k": QWEN3_THINK_TOP_K,
                    "min_p": QWEN3_THINK_MIN_P,
                }
            else:
                return {
                    "temperature": QWEN3_TEMPERATURE,
                    "top_p": QWEN3_TOP_P,
                    "top_k": QWEN3_TOP_K,
                    "min_p": QWEN3_MIN_P,
                }
        else:
            raise ValueError(f"No default values specified for {self.model_id}")

    def stream_llm_response(
        self, messages: list[Message], think: bool = False
    ) -> Generator:
        # Instantiate buffers
        if think:
            thinking_buffer = gr.ChatMessage(
                content="",
                role="assistant",
                metadata={"title": "Thinking", "status": "pending"},
            )
        content_buffer = gr.ChatMessage(content="", role="assistant")
        tool_calls = []

        # Call the LLM
        try:
            llm_stream = chat(
                model=self.model_id,
                messages=messages,
                tools=self.tools if self.is_tool_calling else None,
                stream=True,
                think=think if self.is_thinking else False,
                options=self._get_sample_params(think),
            )
            for chunk in llm_stream:
                # Handle thinking tokens
                if chunk.message.thinking:
                    thinking_buffer.content += chunk.message.thinking
                elif think and thinking_buffer.content:
                    thinking_buffer.metadata["status"] = "done"

                # Handle content tokens
                if chunk.message.content:
                    content_buffer.content += chunk.message.content

                # Handle tool calls
                if chunk.message.tool_calls:
                    for t in chunk.message.tool_calls:
                        tool_name = t.function.name
                        tool_args = t.function.arguments
                        tool_calls.append((tool_name, tool_args))

                elements = [thinking_buffer] if think else []
                if content_buffer.content:
                    elements.append(content_buffer)
                yield elements, tool_calls

        except Exception as e:
            self.logger.error(f"Error in LLM call: {e}")
            yield f"Error in LLM call with {messages}: {e}", []
