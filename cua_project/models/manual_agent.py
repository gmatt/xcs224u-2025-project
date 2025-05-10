import base64
from logging import Logger
from typing import Optional

import ollama
import openai.types.responses
import pyautogui
from openai import OpenAI

from cua_project.models.base_agent import Action, BaseAgent, Observation


class ManualAgent(BaseAgent):
    """
    Agent for experiments that asks an llm what to do based on a screenshot and the
    instruction, but I still need to manually perform the actions, similar to the
    experiments in https://arxiv.org/abs/2311.07562.
    """

    def __init__(
        self,
        model: str,
        *args,
        **kwargs,
    ):
        self.model = model

    action_space = "pyautogui"

    def reset(self, logger: Optional[Logger] = None) -> None:
        pass

    def predict(self, instruction: str, obs: Observation) -> tuple[str, list[Action]]:
        input_text = f"My goal is the following: {instruction}\nI see this screen. What should I do next?"
        print(input_text)
        if self.model.startswith("gpt-"):
            if "Cancel" == pyautogui.confirm(
                text="Send request to OpenAI?",
                buttons=["OK", "Cancel"],
            ):
                raise KeyboardInterrupt
            client = OpenAI()
            response = client.responses.create(
                model="gpt-4.1-nano",
                input=[
                    openai.types.responses.EasyInputMessageParam(
                        role="user",
                        content=[
                            openai.types.responses.ResponseInputTextParam(
                                type="input_text",
                                text=input_text,
                            ),
                            openai.types.responses.ResponseInputImageParam(
                                type="input_image",
                                image_url=f"data:image/png;base64,{base64.b64encode(obs['screenshot']).decode()}",
                                detail="auto",
                            ),
                        ],
                    )
                ],
            )
            response_text = response.output_text
        elif self.model == "ollama":
            response = ollama.chat(
                model="gemma3:4b",
                messages=[
                    ollama.Message(
                        role="user",
                        content=input_text,
                        images=[
                            ollama.Image(
                                value=obs["screenshot"],
                            ),
                        ],
                    )
                ],
            )
            response_text = response.message.content
        else:
            raise ValueError()

        print(response_text)
        button_captions: dict[str, Action] = {
            "Done, what's next?": "WAIT",
            "Task done.": "DONE",
        }
        action = button_captions[
            pyautogui.confirm(text=response_text, buttons=button_captions.keys())
        ]
        return "", [action]
