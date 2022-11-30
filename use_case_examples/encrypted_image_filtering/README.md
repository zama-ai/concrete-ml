---
title: Image Filtering on Encrypted Images using FHE
emoji: 🥷💬
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 3.2
app_file: app.py
pinned: true
tags: [FHE, PPML, privacy, privacy preserving machine learning, homomorphic encryption,
  security]
python_version: 3.8.15
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Image filtering using FHE

## Running the application on your machine

In this directory, ie `encrypted_image_filtering`, you can do the following steps.

### Do once

First, create a virtual env and activate it:

<!--pytest-codeblocks:skip-->

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, install required packages:

<!--pytest-codeblocks:skip-->

```bash
pip3 install -U pip wheel setuptools --ignore-installed
pip3 install -r requirements.txt --ignore-installed
```

If not on Linux, or if you want to compile the FHE filters by yourself:

<!--pytest-codeblocks:skip-->

```bash
python3 compile.py
```

Check it finish well (with a "Done!").

It is also possible to manually add some new filters in `filters.py`. Yet, in order to be able to use
them interactively in the app, you first need to update the `AVAILABLE_FILTERS` list found in `common.py`
and then compile them by running :

<!--pytest-codeblocks:skip-->

```bash
python3 generate_dev_filters.py
```

## Run the following steps each time you relaunch the application

In a terminal (Tab 1, the server side):

<!--pytest-codeblocks:skip-->

```bash
source .venv/bin/activate
uvicorn server:app
```

Then, in another terminal (Tab 2, the client side):

<!--pytest-codeblocks:skip-->

```bash
source .venv/bin/activate
python3 app.py
```

## Interacting with the application

Open the given URL link (search for a line like `Running on local URL:  http://127.0.0.1:8888/` in your Terminal 2).
