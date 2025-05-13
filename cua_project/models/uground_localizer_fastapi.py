import ast
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from matplotlib import pyplot as plt, rcParams
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)

MIN_PIXELS = 802816
MAX_PIXELS = 1806336

model: Optional[Qwen2VLForConditionalGeneration]
processor: Optional[Qwen2VLProcessor]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "osunlp/UGround-V1-2B",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(
        "osunlp/UGround-V1-2B", min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    yield


app = FastAPI(lifespan=lifespan)


class ResponseModel(BaseModel):
    x: float
    y: float


@app.post("/localize")
async def localize(
    label: str = Form(...),
    image: UploadFile = File(...),
) -> ResponseModel:
    with TemporaryDirectory() as tempdir:
        temp_file = Path(tempdir) / "temp.png"
        temp_file.write_bytes(await image.read())
        pil_image = Image.open(temp_file)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a very helpful assistant"},
                    {
                        "type": "image",
                        "image": str(temp_file),
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    },
                    {
                        "type": "text",
                        "text": f"""Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
  - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
  - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
  - Your answer should be a single string (x, y) corresponding to the point of the interest.
  Description: {label}
  Answer:""",
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=128, temperature=0)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        point = ast.literal_eval(output_text)
        x, y = round(point[0] / 1000 * pil_image.width), round(
            point[1] / 1000 * pil_image.height
        )
        coordinates = ResponseModel(
            x=x,
            y=y,
        )

        rcParams["figure.dpi"] = 300
        plt.imshow(pil_image)
        plt.plot(coordinates.x, coordinates.y, marker="+", markersize=20, c="red")
        plt.show()

        return coordinates
