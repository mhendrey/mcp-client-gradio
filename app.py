import asyncio
import json
from time import sleep
from typing import List, Dict, Any, Union

import gradio as gr
from gradio.components.multimodal_textbox import MultimodalValue
from fastmcp import Client
from ollama import chat


mcp_config = {
    "mcpServers": {
        # Local server running via stdio
        "blackforest": {"command": "python", "args": ["-m", "mcp_server_blackforest"]},
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
    }
}

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
HTML_PREFIX = "<!DOCTYPE html>\n<html>\n<body>\n<blockquote>"
HTML_SUFFIX = "\n</blockquote>\n</body>\n</html>"


class MCPClientWrapper:
    def __init__(self):
        self.client = Client(mcp_config)
        self.tools = loop.run_until_complete(self.list_tools())

        # This is an alias for 30b-a3b tag that allows think=True
        self.model_id = "qwen3:30b"

    async def list_tools(self) -> List[Dict]:
        async with self.client:
            print(f"Client connected: {self.client.is_connected()}")
            tools = await self.client.list_tools()
            print(f"Available tools: {[tool.name for tool in tools]}")
            # Convert them to format consumable by ollama
            tools = [{"type": "function", "function": t.model_dump()} for t in tools]

        return tools

    def llm_call(
        self,
        message_text: str,
        history: List[Union[gr.ChatMessage, dict[str, Any]]],
        think: bool = False,
        show_thinking: bool = False,
        tool_calls: list = [],
    ):
        messages = []
        for msg in history:
            if isinstance(msg, gr.ChatMessage):
                role, content = msg.role, msg.content
            elif isinstance(msg, dict):
                try:
                    role, content = msg["role"], msg["content"]
                except Exception as exc:
                    raise ValueError(
                        f"{msg.keys()} missing 'role' and/or 'content' keys"
                    )
            else:
                raise ValueError(
                    f"msg in history is a {type(msg)}. Must be dict or ChatMessage"
                )

            # Skip over files which appear as (f"{file_path"})
            if isinstance(content, tuple):
                continue
            # Skip over Component content
            elif isinstance(content, gr.Component):
                continue

            if role in ["user", "assistant", "system", "developer"]:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message_text})

        # Send to the LLM
        #print(f"Sending to LLM:\n{messages}")
        llm_stream = chat(
            model=self.model_id,
            messages=messages,
            tools=self.tools,
            stream=True,
            think=think,
            options={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
        )

        content_buffer = ""
        thinking_buffer = ""
        started_thinking = False
        for chunk in llm_stream:
            if chunk.message.thinking:
                if not started_thinking:
                    started_thinking = True
                    thinking_buffer += "Thinking...\n\n"
                    yield thinking_buffer
                thinking_buffer += chunk.message.thinking
                if show_thinking:
                    yield gr.HTML(f"{HTML_PREFIX}{thinking_buffer}{HTML_SUFFIX}")
            if chunk.message.content:
                content_buffer += chunk.message.content
                if show_thinking:
                    yield [
                        gr.HTML(f"{HTML_PREFIX}{thinking_buffer}{HTML_SUFFIX}"),
                        content_buffer,
                    ]
                else:
                    yield content_buffer
            if chunk.message.tool_calls:
                for t in chunk.message.tool_calls:
                    tool_name = t.function.name
                    tool_args = t.function.arguments
                    # Prompt with user's exact message
                    if tool_name == "blackforest_generate_image" and not think:
                        tool_args["prompt"] = message_text
                    tool_calls.append((tool_name, tool_args))

    def process_message(
        self,
        message: MultimodalValue,
        history: List[Union[gr.ChatMessage, Dict[str, Any]]],
        think: bool = True,
        show_thinking: bool = False,
    ):
        message_text = message.get("text", "")
        # Skipping the handling of these for now
        # TODO Add in markitdown to handle files too
        # message_files = message.get("files", [])
        tool_calls = []
        chunk = None

        try:
            for chunk in self.llm_call(
                message_text, history, think, show_thinking, tool_calls
            ):
                yield chunk
        except Exception as e:
            yield f"Error doing initial LLM call: {e}"

        if tool_calls:
            for tool_name, tool_args in tool_calls:
                tool_feedback_msg = f"Using {tool_name} tool..."
                yield tool_feedback_msg
                try:
                    result = loop.run_until_complete(
                        self._execute_tool_call(tool_name, tool_args)
                    )
                    print(f"{tool_name} with {tool_args} returns:\n\n{result}")
                    tool_feedback_msg += "tool execution completed"
                    yield tool_feedback_msg
                except Exception as e:
                    yield f"Error executing tool {tool_name} with {tool_args}: {e}"
                if tool_name == "blackforest_generate_image":
                    result = json.loads(result)
                    if result["type"] == "image":
                        yield [
                            gr.Image(result["tmp_file"]),
                            result["message"],
                        ]
                    else:
                        yield result["message"]
                elif tool_name == "fetch_fetch":
                    # Send the original prompt with the now attached text to the LLM again
                    new_prompt = f"{message_text}\n\n<website>{result}</website>"
                    try:
                        for chunk in self.llm_call(
                            new_prompt, [], think, show_thinking, tool_calls
                        ):
                            yield chunk
                    except Exception as e:
                        yield f"Error doing initial LLM call: {e}"

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
            gr.Checkbox(label="Think Mode", show_label=True, value=True),
            gr.Checkbox(label="Show Thinking", show_label=True, value=False),
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
