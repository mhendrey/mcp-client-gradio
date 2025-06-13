import gradio as gr
import logging
from magika import Magika
from ollama import chat, Message
from typing import Any, Generator, Union
import yaml


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
        system_prompt: str,
    ) -> list[Message]:
        """Build messages array to give to Ollama LLM."""
        messages = []
        # Appending system_prompt to the beginning if not already there
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

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
                            message.images = [content[0]]
                        else:
                            message.images.append(content[0])
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
            user_message.images = [f for f in current_images]
        messages.append(user_message)

        return messages


class LLM_Client:
    """Handles calls to the LLM"""

    def __init__(
        self,
        model_id,
        tools: list = [],
    ):
        self.set_model_id(model_id)
        self.tools = tools
        self.models_config = yaml.safe_load(open("config.yaml"))["models"]
        self.logger = logging.getLogger(__name__)

    def set_model_id(self, model_id):
        self.model_id = model_id

    def stream_llm_response(
        self, messages: list[Message], think: bool = False
    ) -> Generator:
        """Send the list of ollama.Message to the LLM

        Parameters
        ----------
        messages : list[Message]
            Includes history plus user's latest prompt.
        think : bool, optional
            Whether to use "thinking" mode if model supports it, by default False

        Yields
        ------
        Generator
            Yields a 2-tuple. First element is a list of gr.ChatMessage which may have
            one or two elements. If two elements, then it is a gr.ChatMessage for the
            thinking content. The second (or only) is the final output of the llm. The
            second element in the tuple is a list of any tool calls. Each element in
            this list is a 2-tuple of the tool_name and the tool_args (dict)
        """
        # Instantiate buffers
        if think:
            thinking_buffer = gr.ChatMessage(
                content="",
                role="assistant",
                metadata={"title": "Thinking", "status": "pending"},
            )
        content_buffer = gr.ChatMessage(content="", role="assistant")
        tool_calls = []

        # Set sampling parameters from config file and call the LLM
        sampling_params = self.models_config[self.model_id].get("sampling_params", None)
        if think:
            sampling_params = self.models_config[self.model_id].get(
                "sampling_params_thinking", sampling_params
            )
        try:
            llm_stream = chat(
                model=self.model_id,
                messages=messages,
                tools=(
                    self.tools
                    if self.models_config[self.model_id]["tool_calling"]
                    else None
                ),
                stream=True,
                think=(
                    think if self.models_config[self.model_id]["thinking"] else False
                ),
                options=sampling_params,
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
