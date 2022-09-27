"""A gradio app. that runs locally (analytics=False and share=False) about sentiment analysis on tweets."""

import gradio as gr
from requests import head
from transformer_vectorizer import TransformerVectorizer
from concrete.ml.deployment import FHEModelClient
import numpy
import os
from pathlib import Path
import requests
import json
import base64
import subprocess
import shutil
import time

subprocess.Popen(["uvicorn", "server:app"])

# Wait 5 sec for the server to start
time.sleep(5)

# Encrypted data limit for the browser to display
# (encrypted data is too large to display in the browser)
ENCRYPTED_DATA_BROWSER_LIMIT = 500
N_USER_KEY_STORED = 20

print("Loading the transformer model...")

# Initialize the transformer vectorizer
transformer_vectorizer = TransformerVectorizer()

def clean_tmp_directory():
    # Allow 20 user keys to be stored.
    # Once that limitation is reached, deleted the oldest.
    list_files = sorted(Path(".fhe_keys/").iterdir(), key=os.path.getmtime)

    user_ids = []
    if len(list_files) > N_USER_KEY_STORED:
        n_files_to_delete = len(list_files) - N_USER_KEY_STORED
        for p in list_files[:n_files_to_delete]:
            user_ids.append(p.name)
            shutil.rmtree(p)

    list_files_tmp = Path("tmp/").iterdir()
    # Delete all files related to user_id
    for file in list_files_tmp:
        for user_id in user_ids:
            if file.name.endswith(f"{user_id}.npy"):
                file.unlink()


def keygen():
    print("Initializing FHEModelClient...")

    # Let's create a user_id
    user_id = numpy.random.randint(0, 2**32)
    fhe_api = FHEModelClient("sentiment_fhe_model/deployment", f".fhe_keys/{user_id}")
    fhe_api.load()


    # Generate a fresh key
    fhe_api.generate_private_and_evaluation_keys(force=True)
    evaluation_key = fhe_api.get_serialized_evaluation_keys()
    size_evaluation_key = len(evaluation_key)

    # Save evaluation_key in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    numpy.save(f"tmp/tmp_evaluation_key_{user_id}.npy", evaluation_key)

    return [list(evaluation_key)[:ENCRYPTED_DATA_BROWSER_LIMIT], size_evaluation_key, user_id]


def encode_quantize_encrypt(text, user_id):
    assert user_id != [], "Please, wait for the creation of FHE keys before trying to encrypt."

    fhe_api = FHEModelClient("sentiment_fhe_model/deployment", f".fhe_keys/{user_id}")
    fhe_api.load()
    encodings = transformer_vectorizer.transform([text])
    quantized_encodings = fhe_api.model.quantize_input(encodings).astype(numpy.uint8)
    encrypted_quantized_encoding = fhe_api.quantize_encrypt_serialize(encodings)

    # Save encrypted_quantized_encoding in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    numpy.save(f"tmp/tmp_encrypted_quantized_encoding_{user_id}.npy", encrypted_quantized_encoding)

    # Compute size
    text_size = len(text.encode())
    encodings_size = len(encodings.tobytes())
    quantized_encoding_size = len(quantized_encodings.tobytes())
    encrypted_quantized_encoding_size = len(encrypted_quantized_encoding)
    encrypted_quantized_encoding_shorten = list(encrypted_quantized_encoding)[:ENCRYPTED_DATA_BROWSER_LIMIT]
    encrypted_quantized_encoding_shorten_hex = ''.join(f'{i:02x}' for i in encrypted_quantized_encoding_shorten)
    return (
        encodings[0],
        quantized_encodings[0],
        encrypted_quantized_encoding_shorten_hex,
        text_size,
        encodings_size,
        quantized_encoding_size,
        encrypted_quantized_encoding_size,
    )


def run_fhe(user_id):
    assert user_id != [], "Please, wait for the creation of FHE keys before trying to predict."

    # Read encrypted_quantized_encoding from the file
    encrypted_quantized_encoding = numpy.load(f"tmp/tmp_encrypted_quantized_encoding_{user_id}.npy")

    # Read evaluation_key from the file
    evaluation_key = numpy.load(f"tmp/tmp_evaluation_key_{user_id}.npy")

    # Use base64 to encode the encodings and evaluation key
    encrypted_quantized_encoding = base64.b64encode(encrypted_quantized_encoding).decode()
    encoded_evaluation_key = base64.b64encode(evaluation_key).decode()

    query = {}
    query["evaluation_key"] = encoded_evaluation_key
    query["encrypted_encoding"] = encrypted_quantized_encoding
    headers = {"Content-type": "application/json"}
    response = requests.post(
        "http://localhost:8000/predict_sentiment", data=json.dumps(query), headers=headers
    )
    encrypted_prediction = base64.b64decode(response.json()["encrypted_prediction"])

    # Save encrypted_prediction in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    numpy.save(f"tmp/tmp_encrypted_prediction_{user_id}.npy", encrypted_prediction)
    encrypted_prediction_shorten = list(encrypted_prediction)[:ENCRYPTED_DATA_BROWSER_LIMIT]
    encrypted_prediction_shorten_hex = ''.join(f'{i:02x}' for i in encrypted_prediction_shorten)
    return encrypted_prediction_shorten_hex


def decrypt_prediction(user_id):
    assert user_id != [], "Please, wait for the creation of FHE keys before trying to decrypt."

    # Read encrypted_prediction from the file
    encrypted_prediction = numpy.load(f"tmp/tmp_encrypted_prediction_{user_id}.npy").tobytes()

    fhe_api = FHEModelClient("sentiment_fhe_model/deployment", f".fhe_keys/{user_id}")
    fhe_api.load()

    # We need to retrieve the private key that matches the client specs (see issue #18)
    fhe_api.generate_private_and_evaluation_keys(force=False)

    predictions = fhe_api.deserialize_decrypt_dequantize(encrypted_prediction)
    return {
        "negative": predictions[0][0],
        "neutral": predictions[0][1],
        "positive": predictions[0][2],
    }


demo = gr.Blocks()


print("Starting the demo...")
with demo:

    # FIXME: to be finalized with marketing
    gr.Markdown(
        """
<h1 align="center">Privacy Preserving Machine Learning with  Fully Homomorphic Encryption.</h1>
<p align="center"><a href="https://github.com/zama-ai/concrete-ml"> Concrete-ML</a> is a Privacy-Preserving Machine Learning (PPML) open-source set of tools which aims to simplify the use of fully homomorphic encryption (FHE) for data scientists.
</p>

"""
    )

    # FIXME: make it smaller and in the middle
    # gr.Image("Zama.svg")

    gr.Markdown(
        """
        <p align="center">
        </p>
        <p align="center">
        </p>
        """
    )
    gr.Markdown(
        """
This short demo explain the concept of Fully Homomorphic Encryption (FHE), here we consider:
- a client, where a tweet is entered, quantized and then encrypted
- a server, which received the encrypted quantized tweet, and analyze it with FHE
Finally, the encrypted sentiment (represented as negative, positive, or neutral) is sent back to the client, which can recover the plain value with her secret key.
"""
    )
    gr.Markdown(
        """
           Finally, the encrypted sentiment (represented as negative, positive, or neutral) is sent back to the client, which can recover the plain value with her secret key.
        """
    )

    gr.Markdown("## Setup")
    gr.Markdown(
    """
The model is previously compiled and sent to the server machine. On the client side, the key must be generated, and the evaluation key must be sent to the server.
Note that,
- the private key is used to encrypt and decrypt the data and shall never be shared.
- the evaluation key is a public key that the server needs to process encrypted data.
"""
    )

    b_gen_key_and_install = gr.Button("Generate the key and send public part to server")

    evaluation_key = gr.Textbox(
        label="Evaluation key (truncated):",
        max_lines=4,
        interactive=False,
    )

    user_id = gr.Textbox(
        label="",
        max_lines=4,
        interactive=False,
        visible=False
    )

    size_evaluation_key = gr.Number(
        label="Size of the evalution key (in bytes):", value=0, interactive=False
    )

    # FIXME: add a picture from marketing with client->server interactions

    gr.Markdown("## Client Side")
    gr.Markdown(
        "Please type a small tweet and press the button: your tweet will be encoded, quantized and encrypted. Then, the encrypted value will be sent to the server."
    )
    text = gr.Textbox(label="Enter a tweet:", value="Zama's Concrete-ML is awesome")
    size_text = gr.Number(label="Size of the text (in bytes):", value=0, interactive=False)
    b_encode_quantize_text = gr.Button(
        "Encode, quantize and encrypt the text with transformer vectorizer, and send to server"
    )

    with gr.Row():
        encoding = gr.Textbox(
            label="Transformer representation:",
            max_lines=4,
            interactive=False,
        )
        quantized_encoding = gr.Textbox(
            label="Quantized transformer representation:", max_lines=4, interactive=False
        )
        encrypted_quantized_encoding = gr.Textbox(
            label="Encrypted quantized transformer representation (truncated):",
            max_lines=4,
            interactive=False,
        )
    with gr.Row():
        size_encoding = gr.Number(label="Size (in bytes):", value=0, interactive=False)
        size_quantized_encoding = gr.Number(label="Size (in bytes):", value=0, interactive=False)
        size_encrypted_quantized_encoding = gr.Number(
            label="Size (in bytes):",
            value=0,
            interactive=False,
        )

    gr.Markdown("## Server Side")
    gr.Markdown(
        "The encrypted value is received by the server. Thanks to the evaluation key and to FHE, the server can compute the (encrypted) prediction directly over encrypted values. Once the computation is finished, the server returns the encrypted prediction to the client."
    )

    b_run_fhe = gr.Button("Run FHE execution there")
    encrypted_prediction = gr.Textbox(
        label="Encrypted prediction (truncated):",
        max_lines=4,
        interactive=False,
    )

    gr.Markdown("## Client Side")
    gr.Markdown(
        "The encrypted sentiment is sent back to client, who can finally decrypt it with her private key. Only the client is aware of the original tweet and the prediction."
    )
    b_decrypt_prediction = gr.Button("Decrypt prediction")

    labels_sentiment = gr.Label(label="Sentiment:")

    # Button for key generation
    b_gen_key_and_install.click(keygen, inputs=[], outputs=[evaluation_key, size_evaluation_key, user_id])

    # Button to quantize and encrypt
    b_encode_quantize_text.click(
        encode_quantize_encrypt,
        inputs=[text, user_id],
        outputs=[
            encoding,
            quantized_encoding,
            encrypted_quantized_encoding,
            size_text,
            size_encoding,
            size_quantized_encoding,
            size_encrypted_quantized_encoding,
        ],
    )

    # Button to send the encodings to the server using post at (localhost:8000/predict_sentiment)
    b_run_fhe.click(run_fhe, inputs=[user_id], outputs=[encrypted_prediction])

    # Button to decrypt the prediction on the client
    b_decrypt_prediction.click(decrypt_prediction, inputs=[user_id], outputs=[labels_sentiment])

demo.launch(share=False)
