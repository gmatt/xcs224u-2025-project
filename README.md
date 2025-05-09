# cua-project

## Setup

```bash
# Optional: Create a Conda environment for OSWorld
# conda create -n osworld python=3.9
# conda activate osworld

# Install required dependencies
pip install -r requirements.txt
```

## Run

```bash
# Need to set pythonpath manually as we don't use any fancy package managers yet.
PYTHONPATH=.:.. python ../cua_project/main.py --model DummyAgent --test_all_meta_path ../cua_project/data/smoketest.json
 ```
