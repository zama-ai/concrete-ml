---
title: Sentiment Analysis on Encrypted Data with FHE
emoji: 🥷💬
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 3.2
app_file: app.py
pinned: true
tags: [FHE, PPML, privacy, privacy preserving machine learning, homomorphic encryption,
  security]
python_version: 3.9
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Sentiment Analysis With FHE

## Running the application on your machine

In this directory, ie `sentiment-analysis-with-transformer`, you can do the following steps.

### Do once

- First, create a virtual env and activate it:

<!--pytest-codeblocks:skip-->

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

- Then, install required packages:

<!--pytest-codeblocks:skip-->

```bash
pip3 install -U pip wheel setuptools --ignore-installed
pip3 install -r requirements.txt --ignore-installed
```

- If not on Linux, or if you want to compile the FHE algorithms by yourself:

<!--pytest-codeblocks:skip-->

```bash
python3 compile.py
```

Check it finish well (with a "Done!").

### Do each time you relaunch the application

- Then, in a terminal Tab 1:

<!--pytest-codeblocks:skip-->

```bash
source .venv/bin/activate
uvicorn server:app
```

Tab 1 will be for the Server side.

- And, in another terminal Tab 2:

<!--pytest-codeblocks:skip-->

```bash
source .venv/bin/activate
python3 app.py
```

Tab 2 will be for the Client side.

## Interacting with the application

Open the given URL link (search for a line like `Running on local URL:  http://127.0.0.1:8888/` in your Terminal 2).

## Training a new model

The notebook SentimentClassification.ipynb provides a way to train a new model.

Before running the notebook, you need to download the data.

<!--pytest-codeblocks:skip-->

```bash
bash download_data.sh
```
