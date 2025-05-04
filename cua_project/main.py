import os
from pathlib import Path

from run import config, test

if __name__ == "__main__":
    # Simple smoke test.

    # Make sure we run the script from the OSWorld directory as cwd affects behavior.
    assert Path(os.getcwd()).name == "OSWorld"

    args = config()
    args.observation_type = "screenshot"
    args.model = "gpt-4-vision-preview"

    # First entry of dataset only.
    test_all_meta = {
        "chrome": [
            "bb5e4c0d-f964-439c-97b6-bdb9747de3f4",
        ],
    }
    test(
        args=args,
        test_all_meta=test_all_meta,
    )
