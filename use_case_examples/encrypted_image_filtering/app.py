"""A local gradio app that filters images using FHE."""

import gradio as gr
import numpy
import os
import requests
import json
import base64
import subprocess
import shutil
import time

from custom_client_server import CustomFHEClient
from common import AVAILABLE_FILTERS, FILTERS_PATH, TMP_PATH, KEYS_PATH, INPUT_SHAPE, REPO_DIR, KEYS_PATH, EXAMPLES

# Uncomment here to have both the server and client in the same terminal
# subprocess.Popen(["uvicorn", "server:app"], cwd=REPO_DIR)

# Wait 5 sec for the server to start
time.sleep(5)


def shorten_bytes_object(bytes_object, limit=500):
    """Shorten the input bytes object to a given length.

    Encrypted data is too large for displaying it in the browser using Gradio. This function 
    provides a shorten representation of it.

    Args:
        bytes_object (bytes): The input to shorten
        limit (int): The length to consider. Default to 500.

    Returns:
        Any: The fitted model.
    
    """
    # Define a shift for better display
    shift = 100
    return bytes_object[shift:limit+shift].hex()


def get_client(user_id, image_filter):
    """Get the client API.

    Args:
        user_id (int): The current user's ID.
        image_filter (str): The filter chosen by the user

    Returns:
        CustomFHEClient: The client API.
    """
    return CustomFHEClient(
        FILTERS_PATH / f"{image_filter}/deployment",
        KEYS_PATH / f"{image_filter}_{user_id}"
    )

def get_file_path(name, user_id, image_filter):
    """Get the correct file path.

    Args:
        name (str): The desired file name.
        user_id (int): The current user's ID.
        image_filter (str): The filter chosen by the user
    
    Returns:
        pathlib.Path: The file path.
    """
    return TMP_PATH / f"tmp_{name}_{image_filter}_{user_id}.npy"

def clean_temporary_files(n_keys=20):
    """Clean keys and encrypted images.

    A maximum of n_keys keys are allowed to be stored. Once this limit is reached, the oldest are
    deleted.

    Args:
        n_keys (int): The maximum number of keys to be stored. Default to 20.

    """
    # Get the oldest files in the key directory 
    list_files = sorted(KEYS_PATH.iterdir(), key=os.path.getmtime)

    # If more than n_keys keys are found, remove the oldest
    user_ids = []
    if len(list_files) > n_keys:
        n_files_to_delete = len(list_files) - n_keys
        for p in list_files[:n_files_to_delete]:
            user_ids.append(p.name)
            shutil.rmtree(p)

    # Get all the encrypted objects in the temporary folder
    list_files_tmp = TMP_PATH.iterdir()

    # Delete all files related to the current user
    for file in list_files_tmp:
        for user_id in user_ids:
            if file.name.endswith(f"{user_id}.npy"):
                file.unlink()

def keygen(image_filter):
    """Generate the private key associated to a filter.

    Args:
        image_filter (str): The current filter to consider.
    
    Returns:
        (user_id, True) (Tuple[int, bool]): The current user's ID and a boolean used for visual display. 

    """
    # Clean temporary files
    clean_temporary_files()

    # Create an ID for the current user
    user_id = numpy.random.randint(0, 2**32)

    # Retrieve the client API
    # Currently, the key generation needs to be done after choosing a filter
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2258
    client = get_client(user_id, image_filter)

    # Generate a private key
    client.generate_private_and_evaluation_keys(force=True)

    # Retrieve the serialized evaluation key. In this case, as circuits are fully leveled, this 
    # evaluation key is empty. However, for software reasons, it is still needed for proper FHE
    # execution
    evaluation_key = client.get_serialized_evaluation_keys()

    # Save evaluation_key in a file as it is too large to pass through regular Gradio buttons (see
    # https://github.com/gradio-app/gradio/issues/1877)
    evaluation_key_path = get_file_path("evaluation_key", user_id, image_filter)
    numpy.save(evaluation_key_path, evaluation_key)

    return (user_id, True)


def encrypt(user_id, input_image, image_filter):
    """Encrypt the given image for a specific user and filter.

    Args:
        user_id (int): The current user's ID.
        input_image (numpy.ndarray): The image to encrypt.
        image_filter (str): The current filter to consider.
    
    Returns:
        (input_image, encrypted_image_short) (Tuple[bytes]): The encrypted image and one of its
        representation.

    """
    if user_id == '':
        raise gr.Error('Please generate the private key first.')

    # Retrieve the client API
    client = get_client(user_id, image_filter)

    # Pre-process, encrypt and serialize the image
    encrypted_image = client.pre_process_encrypt_serialize(input_image)

    # Save encrypted_image in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    encrypted_input_path = get_file_path("encrypted_image", user_id, image_filter)
    numpy.save(encrypted_input_path, encrypted_image)

    # Create a truncated version of the encrypted image for display
    encrypted_image_short = shorten_bytes_object(encrypted_image)

    return (input_image, encrypted_image_short)


def run_fhe(user_id, image_filter):
    """Apply the filter on the encrypted image using FHE.

    Args:
        user_id (int): The current user's ID.
        image_filter (str): The current filter to consider.
    
    Returns:
        encrypted_output_image_short (bytes): A representation of the encrypted result.

    """
    # Get the evaluation key path
    evaluation_key_path = get_file_path("evaluation_key", user_id, image_filter)

    if user_id == '' or not evaluation_key_path.is_file():
        raise gr.Error('Please generate the private key first.')
    
    encrypted_input_path = get_file_path("encrypted_image", user_id, image_filter)

    if not encrypted_input_path.is_file():
        raise gr.Error('Please generate the private key and then encrypt an image first.')

    # Load the encrypted image 
    encrypted_image = numpy.load(encrypted_input_path)

    # Load the evaluation key
    evaluation_key = numpy.load(evaluation_key_path)

    # Use base64 to encode the encrypted image and evaluation key. This is used to simulate a real
    # client-server interface
    # Investigate if is possible to make this encoding/decoding process faster 
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2257
    encrypted_encoding = base64.b64encode(encrypted_image).decode()
    encoded_evaluation_key = base64.b64encode(evaluation_key).decode()

    # Define the post query
    query = {}
    query["filter"] = image_filter
    query["evaluation_key"] = encoded_evaluation_key
    query["encrypted_image"] = encrypted_encoding

    # Call the server to run the FHE executions on the encrypted image 
    headers = {"Content-type": "application/json"}
    response = requests.post(
        "http://localhost:8000/filter_image", data=json.dumps(query), headers=headers
    )

    # Retrieve and decode the server's encrypted output
    encrypted_output_image = base64.b64decode(response.json()["encrypted_output_image"])

    # Save the encrypted output in a file as it is too large to pass through regular Gradio
    # buttons (see https://github.com/gradio-app/gradio/issues/1877)
    encrypted_output_path = get_file_path("encrypted_output", user_id, image_filter)
    numpy.save(encrypted_output_path, encrypted_output_image)
    
    # Create a truncated version of the encrypted output for display
    encrypted_output_image_short = shorten_bytes_object(encrypted_output_image)

    return encrypted_output_image_short


def decrypt_output(user_id, image_filter):
    """Decrypt the result.

    Args:
        user_id (int): The current user's ID.
        image_filter (str): The current filter to consider.
    
    Returns:
        output_image (numpy.ndarray): The decrypted output.

    """
    if user_id == '':
        raise gr.Error('Please generate the private key first.')

    # Get the encrypted output path
    encrypted_output_path = get_file_path("encrypted_output", user_id, image_filter)

    if not encrypted_output_path.is_file():
        raise gr.Error('Please run the FHE execution first.')

    # Load (in bytes) the encrypted output
    encrypted_output_image = numpy.load(encrypted_output_path).tobytes()

    # Retrieve the client API
    client = get_client(user_id, image_filter)

    # Deserialize, decrypt and post-process the encrypted output 
    output_image = client.deserialize_decrypt_post_process(encrypted_output_image)

    return output_image


demo = gr.Blocks(show_error=True)


print("Starting the demo...")
with demo:
    gr.Markdown(
        """
        <p align="center">
        </p>
        <p align="center">
        </p>
        """
    )

    gr.Markdown("## Client side")
    gr.Markdown(
        f"Upload an image. It will automatically be resized to shape ({INPUT_SHAPE[0]}x{INPUT_SHAPE[1]})."
        "The image is however displayed using its original resolution."
    )
    with gr.Row():
        input_image = gr.Image(label="Upload an image here.", shape=INPUT_SHAPE, source="upload", interactive=True)

        examples = gr.Examples(examples=EXAMPLES, inputs=[input_image], examples_per_page=5, label="Examples to use.")

    gr.Markdown("Choose your filter")
    image_filter = gr.Dropdown(choices=AVAILABLE_FILTERS, value="inverted", label="Choose your filter", interactive=True)

    gr.Markdown("### Notes")
    gr.Markdown(
        """
        - The private key is used to encrypt and decrypt the data and shall never be shared.
        - No public key are required for these filter operators.
        """
        )

    keygen_button = gr.Button("Generate the private key.")

    keygen_checkbox = gr.Checkbox(label="Private key generated.", interactive=False)

    encrypt_button = gr.Button(
        "Encrypt the image using FHE."
    )

    user_id = gr.Textbox(
        label="",
        max_lines=4,
        interactive=False,
        visible=False
    )

    # Maybe display an image representation
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2265
    encrypted_image = gr.Textbox(label="Encrypted image representation:", max_lines=4, interactive=False) 

    gr.Markdown("## Server side")
    gr.Markdown(
        "The encrypted value is received by the server. The server can then compute the filter "
        "directly over encrypted values. Once the computation is finished, the server returns "
        "the encrypted results to the client."
    )

    execute_fhe_button = gr.Button("Run FHE execution")

    # Maybe display an image representation
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2265
    encrypted_output_image = gr.Textbox(label="Encrypted output image representation:", max_lines=4, interactive=False) 

    gr.Markdown("## Client side")
    gr.Markdown(
        "The encrypted output is sent back to client, who can finally decrypt it with its "
        "private key. Only the client is aware of the original image and its transformed version."
    )

    decrypt_button = gr.Button("Decrypt the output")

    # Final input vs output display
    with gr.Row():
        original_image = gr.Image(input_image.value, label=f"Input image ({INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}):", interactive=False) 
        original_image.style(height=256, width=256)

        output_image = gr.Image(label=f"Output image ({INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}):", interactive=False) 
        output_image.style(height=256, width=256)

    # Button to generate the private key
    keygen_button.click(
        keygen,
        inputs=[image_filter],
        outputs=[user_id, keygen_checkbox],
    )

    # Button to encrypt inputs on the client side
    encrypt_button.click(
        encrypt,
        inputs=[user_id, input_image, image_filter],
        outputs=[
            original_image,
            encrypted_image,
        ],
    )  

    # Button to send the encodings to the server using post at (localhost:8000/transform_image)
    execute_fhe_button.click(run_fhe, inputs=[user_id, image_filter], outputs=[encrypted_output_image])

    # Button to decrypt the output on the client side
    decrypt_button.click(decrypt_output, inputs=[user_id, image_filter], outputs=[output_image])

    gr.Markdown(
        "The app was built with [Concrete-ML](https://github.com/zama-ai/concrete-ml), a "
        "Privacy-Preserving Machine Learning (PPML) open-source set of tools by [Zama](https://zama.ai/). "
        "Try it yourself and don't forget to star on Github &#11088;."
    )

demo.launch(share=False)
