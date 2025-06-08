from fastmcp.server import FastMCP
from gradio_client import Client
import io
import json
from pydantic import Field
import sys
from typing import Annotated

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

mcp = FastMCP("Huggingface_spaces_image_display")


@mcp.tool()
async def generate_image(
    prompt: str,
    width: int = 512,
    height: int = 512,
    # prompt: Annotated[str, Field(description="Prompt describing the image to generate")],
    # width: Annotated[int, Field(description="Image width in pixels. Default is 512")] = 512,
    # height: Annotated[int, Field(description="Image height in pixels. Default is 512")] = 512,
) -> dict:
    client = Client("black-forest-labs/FLUX.1-schnell")
    try:
        # Returns a tuple ("/tmp/gradio/{hash}/image.webp", seed:int)
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=width,
            height=height,
            num_inference_steps=4,
            api_name="/infer",
        )

        if isinstance(result, tuple) and len(result) == 2:
            image_file = result[0]
            return {
                "type": "image",
                "tmp_file": image_file,
                "message": f"Generated image for prompt: {prompt}",
            }
        else:
            return {
                "type": "error",
                "message": f"Failed to generate image. {result=:}",
            }
    except Exception as exc:
        return {
            "type": "error",
            "message": f"Error generating image: {exc}",
        }

    # @mcp.tool()
    # async def generate_image(prompt: str, width: int = 512, height: int = 512) -> str:
    """Generate an image using BlackForest Labs FLUX.1-schnell model.

    Parameters
    ----------
    prompt : str
        Text prompt describing the image to generate
    width : int, optional
        Image width, by default 512
    height : int, optional
        Image height, by default 512

    Returns
    -------
    str
        JSON string with keys: type, message, image (if successful)
    """


#    client = Client("black-forest-labs/FLUX.1-schnell")
#    try:
#        # Returns a tuple ("/tmp/gradio/{hash}/image.webp", seed:int)
#        result = client.predict(
#            prompt=prompt,
#            seed=0,
#            randomize_seed=True,
#            width=width,
#            height=height,
#            num_inference_steps=4,
#            api_name="/infer",
#        )

#        if isinstance(result, tuple) and len(result) == 2:
#            image_file = result[0]
#            return json.dumps(
#                {
#                    "type": "image",
#                    "tmp_file": image_file,
#                    "message": f"Generated image for prompt: {prompt}",
#                }
#            )
#        else:
#            return json.dumps(
#                {
#                    "type": "error",
#                    "message": f"Failed to generate image. {result=:}",
#                }
#            )
#    except Exception as exc:
#        return json.dumps(
#            {
#                "type": "error",
#                "message": f"Error generating image: {exc}",
#            }
#        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
