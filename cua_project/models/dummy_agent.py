from logging import Logger
from typing import Optional

from cua_project.models.base_agent import Action, BaseAgent, Observation


class DummyAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        pass

    action_space = "pyautogui"

    def reset(self, logger: Optional[Logger] = None) -> None:
        pass

    def predict(self, instruction: str, obs: Observation) -> tuple[str, list[Action]]:
        # return "", ["pyautogui.click(960, 540)"]
        return "", ["DONE"]
