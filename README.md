# cua-project

This repo contains some experiments on the OSWorld benchmark.

Although the code runs and the experiments work, the model itself is incomplete and is
not really useful outside of these specific experiments.

## Setup

I did a pip freeze on the requirements for reproducibility, but it includes some junk
not needed for execution, like jupyter.

You're probably better off just installing `OSWorld/requirements.txt` and
`cua_project/requirements.txt` instead of the command below.

```bash
# Optional: Create a Conda environment for OSWorld
# conda create -n osworld python=3.9
# conda activate osworld

# Install required dependencies
pip install -r requirements.txt

git submodule init
git submodule update
# Or just clone the OSWorld repo and put it inside this dir.
```

## Run

Test to see if the setup works.

```bash
# Need to set pythonpath manually as we don't use any fancy package managers yet.
cd OSWorld
PYTHONPATH=.:.. python ../cua_project/main.py --model DummyAgent --test_all_meta_path ../cua_project/data/smoketest.json
 ```

Run the whole evaluation. (Needs `OPENAI_API_KEY` envvar to be set and will probably cost you money.)

```bash
cd OSWorld
PYTHONPATH=.:.. python ../cua_project/main.py --model MainAgent --test_all_meta_path ./evaluation_examples/test_small.json
```

The model also needs the localization api ran in a different process either on localhost, or on a remote server.

When running on a remote server, update the remote URL in `localizer_client.py`.

```bash
uvicorn cua_project.models.uground_localizer_fastapi:app --host 0.0.0.0 --port 8001
```
