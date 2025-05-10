from abc import ABC, abstractmethod
from logging import Logger
from typing import Literal, Optional, TypedDict, Union


class Observation(TypedDict):
    screenshot: Optional[bytes]
    accessibility_tree: Optional[str]
    terminal: Optional[str]
    instruction: str


Action = Union[Literal["WAIT", "DONE", "FAIL"], str]


class BaseAgent(ABC):
    """
    Copied form OSWorld/mm_agents/agent.py
    """

    @abstractmethod
    def __init__(
        self,
        platform: Literal["ubuntu", "windows"] = "ubuntu",
        model: str = "gpt-4-vision-preview",
        max_tokens: int = 1500,
        top_p: float = 0.9,
        temperature: float = 0.5,
        action_space: Literal["computer_13", "pyautogui"] = "pyautogui",
        observation_type: Literal[
            "screenshot",
            "a11y_tree",
            "screenshot_a11y_tree",
            "som",
        ] = "screenshot",
        max_trajectory_length: int = 3,
        a11y_tree_max_tokens: int = 10000,
    ):
        pass

    @abstractmethod
    def reset(
        self,
        logger: Optional[Logger] = None,
    ) -> None:
        pass

    @abstractmethod
    def predict(
        self,
        instruction: str,
        obs: Observation,
    ) -> tuple[str, list[Action]]:
        pass

    @property
    @abstractmethod
    def action_space(self) -> Literal["pyautogui"]:
        pass
