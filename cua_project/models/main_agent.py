import base64
import re
from io import BytesIO
from logging import Logger
from typing import Optional

import ollama
import openai.types.responses
import pyautogui
from matplotlib import pyplot as plt, rcParams
from openai import OpenAI

from cua_project.models.base_agent import Action, BaseAgent, Observation
from cua_project.util.localizer_client import LocalizerClient


class MainAgent(BaseAgent):
    def __init__(
        self,
        model: str,
        observation_type: str,
        temperature: float = 0.0,
        *args,
        **kwargs,
    ):
        self.history: list[str] = []
        self.model = model
        self.temperature = temperature

    action_space = "pyautogui"

    def ask_llm(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        model: Optional[str] = None,
        chat_history: Optional[list[str]] = None,
    ):
        """

        Args:
            prompt:
            image:
            model:
            chat_history: List of messages in the following order: query, response, query...

        Returns:

        """
        if model is None:
            model = self.model
        if chat_history is None:
            chat_history = []

        if model.startswith("gpt-"):
            if "Cancel" == pyautogui.confirm(
                title=f"Send the following request to OpenAI? image={bool(image)}",
                text=prompt,
                buttons=["OK", "Cancel"],
            ):
                raise KeyboardInterrupt
            client = OpenAI()
            response = client.responses.create(
                model=model,
                temperature=self.temperature,
                input=[
                    *(
                        message
                        for query, response in zip(
                            chat_history[::2], chat_history[1::2]
                        )
                        for message in (
                            openai.types.responses.ResponseInputImageParam(
                                role="user",
                                content=[
                                    openai.types.responses.ResponseInputTextParam(
                                        type="input_text",
                                        text=query,
                                    ),
                                ],
                            ),
                            openai.types.responses.ResponseInputImageParam(
                                role="assistant",
                                content=[
                                    openai.types.responses.ResponseInputTextParam(
                                        type="output_text",
                                        text=response,
                                    ),
                                ],
                            ),
                        )
                    ),
                    openai.types.responses.EasyInputMessageParam(
                        role="user",
                        content=[
                            openai.types.responses.ResponseInputTextParam(
                                type="input_text",
                                text=prompt,
                            ),
                            *(
                                [
                                    openai.types.responses.ResponseInputImageParam(
                                        type="input_image",
                                        image_url=f"data:image/png;base64,{base64.b64encode(image).decode()}",
                                        detail="auto",
                                    )
                                ]
                                if image
                                else []
                            ),
                        ],
                    ),
                ],
            )
            response_text = response.output_text
        elif model == "ollama":
            response = ollama.chat(
                model="gemma3:4b",
                temperature=self.temperature,
                messages=[
                    ollama.Message(
                        role="user",
                        content=prompt,
                        images=(
                            [
                                ollama.Image(
                                    value=image,
                                ),
                            ]
                            if image
                            else None
                        ),
                    )
                ],
            )
            response_text = response.message.content
        else:
            raise ValueError()
        return response_text

    def reset(self, logger: Optional[Logger] = None) -> None:
        self.history = []

    def predict(self, instruction: str, obs: Observation) -> tuple[str, list[Action]]:
        screenshot = obs["screenshot"]
        rcParams["figure.dpi"] = 300
        plt.imshow(plt.imread(BytesIO(screenshot)))
        plt.show()

        if not self.history:
            prompt = f"My goal is the following: {instruction}\nI see this screen. What should I do next?"
            print(prompt)
        else:
            prompt = (
                "Ok, I did one step and now I see this screen. What should I do next?"
            )
            print([*self.history, prompt])
        self.history.append(prompt)
        response_text = self.ask_llm(
            prompt=prompt,
            image=screenshot,
            chat_history=self.history,
        )
        self.history.append(response_text)
        print(response_text)

        prompt = f"""Take the first step of the following instructions. If it's a click, answer 'CLICK ' followed by a precise description where to click,
otherwise if it's a scroll, type, hotkey, etc, answer with a pyautogui code, like 'pyautogui.write(...)'.
---
{response_text}"""
        action_text = self.ask_llm(prompt)
        print(action_text)
        if action_text.strip().lower().startswith("click"):
            coordinates = LocalizerClient().localize(
                image=screenshot,
                label=re.sub("click ", "", action_text, flags=re.IGNORECASE),
            )
            action = f"pyautogui.click({coordinates['x']}, {coordinates['y']})"
        else:
            action = action_text

        if "Yes" == pyautogui.confirm(text="Done?", buttons=["Yes", "No"]):
            action = "DONE"

        return "", [action]
