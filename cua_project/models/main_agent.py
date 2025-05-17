import base64
import re
from logging import Logger
from typing import Optional

import ollama
import openai.types.responses
import pyautogui
from openai import OpenAI

from cua_project.models.base_agent import Action, BaseAgent, Observation
from cua_project.util.localizer_client import LocalizerClient

USE_HISTORY = False


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
        choice = pyautogui.confirm(text="Done?", buttons=["Yes", "No", "Abort"])
        if choice == "Yes":
            return "DONE", ["DONE"]
        elif choice == "Abort":
            return "FAIL", ["FAIL"]

        screenshot = obs["screenshot"]
        # rcParams["figure.dpi"] = 300
        # plt.imshow(plt.imread(BytesIO(screenshot)))
        # plt.show()

        if not USE_HISTORY or not self.history:
            prompt = (
                f"My goal is the following: {instruction}\n"
                "I made some progress and currently see this screen. What should I do next?\n"
                # "Please pay attention to every requirement.\n"
                # "Leave everything unspecified as-is.\n"
                # "Do not include steps that are already done.\n"
                # "Do not include steps to verify anything, leave them implicit.\n"
                # "Prefer hotkeys over clicks where it makes sense.\n"
                # "In Chrome, if the bookmarks bar is not visible, enable it first.\n"
            )
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
            chat_history=self.history if USE_HISTORY else None,
        )
        self.history.append(response_text)
        print(response_text)

        prompt = (
            # "Take the first actionable step (that's not already done) of the following instructions.\n"
            "Take the first step of the following instructions.\n"
            # "If a step seems already done, like the menu is already open, take the next one instead.\n"
            # "Ignore instructions like 'verify' or with comments like 'it looks like it's already open'.\n"
            "If it's a click, answer 'CLICK ' followed by a precise description where to click,\n"
            "otherwise if it's a scroll, type, hotkey, etc, answer with a pyautogui code, like 'pyautogui.write(...)'.\n"
            # "If the answer mentions that the task is done, answer 'DONE'.\n"
            "---\n"
            f"{response_text}"
        )
        action_text = self.ask_llm(
            prompt=prompt,
            # A small model is enough for this independent of the 'main' model.
            model="gpt-4.1-nano-2025-04-14",
        )
        print(action_text)
        if action_text.strip().lower().startswith("click"):
            coordinates = LocalizerClient().localize(
                image=screenshot,
                label=re.sub("click ", "", action_text, flags=re.IGNORECASE),
            )
            action = f"pyautogui.click({coordinates['x']}, {coordinates['y']})"
        else:
            action = action_text

        return action_text, [action]
