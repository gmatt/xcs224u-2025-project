import argparse
import json
import os
import re
import unittest.mock
from pathlib import Path

from mm_agents.agent import PromptAgent
from run import config, test


def main(
    agent_class_name: str,
    dataset_json_path: Path,
):
    # Make sure we run the script from the OSWorld directory as cwd affects behavior.
    assert Path(os.getcwd()).name == "OSWorld"

    osworld_args = config()
    osworld_args.observation_type = "screenshot"
    osworld_args.temperature = 0.0
    osworld_args.sleep_after_execution = 5.0

    test_all_meta = json.loads(dataset_json_path.read_text())

    if agent_class_name == "PromptAgent":
        # Default one from OSWorld.
        agent_class = PromptAgent
    else:
        camel_case = re.sub(r"([a-z])([A-Z])", r"\1_\2", agent_class_name).lower()
        agent_class = getattr(
            __import__("cua_project.models." + camel_case, fromlist=[agent_class_name]),
            agent_class_name,
        )

    with unittest.mock.patch("run.PromptAgent") as mock:
        # There is no config option for this inside OSWorld, so we need to mock.
        mock.side_effect = agent_class
        test(
            args=osworld_args,
            test_all_meta=test_all_meta,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Agent class name.",
        required=True,
    )
    parser.add_argument(
        "--test_all_meta_path",
        help="Dataset JSON path.",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    main(
        agent_class_name=args.model,
        dataset_json_path=args.test_all_meta_path,
    )
