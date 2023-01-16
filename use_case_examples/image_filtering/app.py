"""A local gradio app that filters images using FHE."""

import os
import shutil
import subprocess
import time

import gradio as gr
import numpy
import requests
from common import (
    AVAILABLE_FILTERS,
    CLIENT_TMP_PATH,
    EXAMPLES,
    FILTERS_PATH,
    INPUT_SHAPE,
    KEYS_PATH,
    REPO_DIR,
    SERVER_URL,
)
from custom_client_server import CustomFHEClient

# Uncomment here to have both the server and client in the same terminal
subprocess.Popen(["uvicorn", "server:app"], cwd=REPO_DIR)
time.sleep(3)


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
    return bytes_object[shift : limit + shift].hex()


def get_client(user_id, image_filter):
    """Get the client API.

    Args:
        user_id (int): The current user's ID.
        image_filter (str): The filter chosen by the user

    Returns:
        CustomFHEClient: The client API.
    """
    return CustomFHEClient(
        FILTERS_PATH / f"{image_filter}/deployment", KEYS_PATH / f"{image_filter}_{user_id}"
    )


def get_client_file_path(name, user_id, image_filter):
    """Get the correct temporary file path for the client.

    Args:
        name (str): The desired file name.
        user_id (int): The current user's ID.
        image_filter (str): The filter chosen by the user

    Returns:
        pathlib.Path: The file path.
    """
    return CLIENT_TMP_PATH / f"{name}_{image_filter}_{user_id}"


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
    list_files_tmp = CLIENT_TMP_PATH.iterdir()

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
    client = get_client(user_id, image_filter)

    # Generate a private key
    client.generate_private_and_evaluation_keys(force=True)

    # Retrieve the serialized evaluation key. In this case, as circuits are fully leveled, this
    # evaluation key is empty. However, for software reasons, it is still needed for proper FHE
    # execution
    evaluation_key = client.get_serialized_evaluation_keys()

    # Save evaluation_key as bytes in a file as it is too large to pass through regular Gradio
    # buttons (see https://github.com/gradio-app/gradio/issues/1877)
    evaluation_key_path = get_client_file_path("evaluation_key", user_id, image_filter)

    with evaluation_key_path.open("wb") as evaluation_key_file:
        evaluation_key_file.write(evaluation_key)

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
    if user_id == "":
        raise gr.Error("Please generate the private key first.")

    if input_image is None:
        raise gr.Error("Please choose an image first.")

    # Retrieve the client API
    client = get_client(user_id, image_filter)

    # Pre-process, encrypt and serialize the image
    encrypted_image = client.pre_process_encrypt_serialize(input_image)

    # Save encrypted_image to bytes in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    encrypted_image_path = get_client_file_path("encrypted_image", user_id, image_filter)

    with encrypted_image_path.open("wb") as encrypted_image_file:
        encrypted_image_file.write(encrypted_image)

    # Create a truncated version of the encrypted image for display
    encrypted_image_short = shorten_bytes_object(encrypted_image)

    return (input_image, encrypted_image_short)


def send_input(user_id, image_filter):
    """Send the encrypted input image as well as the evaluation key to the server.

    Args:
        user_id (int): The current user's ID.
        image_filter (str): The current filter to consider.
    """
    # Get the evaluation key path
    evaluation_key_path = get_client_file_path("evaluation_key", user_id, image_filter)

    if user_id == "" or not evaluation_key_path.is_file():
        raise gr.Error("Please generate the private key first.")

    encrypted_input_path = get_client_file_path("encrypted_image", user_id, image_filter)

    if not encrypted_input_path.is_file():
        raise gr.Error("Please generate the private key and then encrypt an image first.")

    # Define the data and files to post
    data = {
        "user_id": user_id,
        "filter": image_filter,
    }

    files = [
        ("files", open(encrypted_input_path, "rb")),
        ("files", open(evaluation_key_path, "rb")),
    ]

    # Send the encrypted input image and evaluation key to the server
    url = SERVER_URL + "send_input"
    with requests.post(
        url=url,
        data=data,
        files=files,
    ) as response:
        return response.ok


def run_fhe(user_id, image_filter):
    """Apply the filter on the encrypted image previously sent using FHE.

    Args:
        user_id (int): The current user's ID.
        image_filter (str): The current filter to consider.
    """
    data = {
        "user_id": user_id,
        "filter": image_filter,
    }

    # Trigger the FHE execution on the encrypted image previously sent
    url = SERVER_URL + "run_fhe"
    with requests.post(
        url=url,
        data=data,
    ) as response:
        if response.ok:
            return response.json()
        else:
            raise gr.Error("Please wait for the input image to be sent to the server.")


def get_output(user_id, image_filter):
    """Retrieve the encrypted output image.

    Args:
        user_id (int): The current user's ID.
        image_filter (str): The current filter to consider.

    Returns:
        encrypted_output_image_short (bytes): A representation of the encrypted result.

    """
    data = {
        "user_id": user_id,
        "filter": image_filter,
    }

    # Retrieve the encrypted output image
    url = SERVER_URL + "get_output"
    with requests.post(
        url=url,
        data=data,
    ) as response:
        if response.ok:
            # Save the encrypted output to bytes in a file as it is too large to pass through regular
            # Gradio buttons (see https://github.com/gradio-app/gradio/issues/1877)
            encrypted_output_path = get_client_file_path("encrypted_output", user_id, image_filter)

            with encrypted_output_path.open("wb") as encrypted_output_file:
                encrypted_output_file.write(response.content)

            # Create a truncated version of the encrypted output for display
            encrypted_output_image_short = shorten_bytes_object(response.content)

            return encrypted_output_image_short
        else:
            raise gr.Error("Please wait for the FHE execution to be completed.")


def decrypt_output(user_id, image_filter):
    """Decrypt the result.

    Args:
        user_id (int): The current user's ID.
        image_filter (str): The current filter to consider.

    Returns:
        (output_image, False, False) ((Tuple[numpy.ndarray, bool, bool]): The decrypted output, as
            well as two booleans used for resetting Gradio checkboxes

    """
    if user_id == "":
        raise gr.Error("Please generate the private key first.")

    # Get the encrypted output path
    encrypted_output_path = get_client_file_path("encrypted_output", user_id, image_filter)

    if not encrypted_output_path.is_file():
        raise gr.Error("Please run the FHE execution first.")

    # Load the encrypted output as bytes
    with encrypted_output_path.open("rb") as encrypted_output_file:
        encrypted_output_image = encrypted_output_file.read()

    # Retrieve the client API
    client = get_client(user_id, image_filter)

    # Deserialize, decrypt and post-process the encrypted output
    output_image = client.deserialize_decrypt_post_process(encrypted_output_image)

    return output_image, False, False


demo = gr.Blocks()


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
        f"Step 1. Upload an image. It will automatically be resized to shape ({INPUT_SHAPE[0]}x{INPUT_SHAPE[1]})."
        "The image is however displayed using its original resolution."
    )
    with gr.Row():
        input_image = gr.Image(
            label="Upload an image here.", shape=INPUT_SHAPE, source="upload", interactive=True
        )

        examples = gr.Examples(
            examples=EXAMPLES, inputs=[input_image], examples_per_page=5, label="Examples to use."
        )

    gr.Markdown("Step 2. Choose your filter")
    image_filter = gr.Dropdown(
        choices=AVAILABLE_FILTERS, value="inverted", label="Choose your filter", interactive=True
    )

    gr.Markdown("### Notes")
    gr.Markdown(
        """
        - The private key is used to encrypt and decrypt the data and shall never be shared.
        - No public key are required for these filter operators.
        """
    )

    with gr.Row():
        keygen_button = gr.Button("Step 3. Generate the private key.")

        keygen_checkbox = gr.Checkbox(label="Private key generated:", interactive=False)

    with gr.Row():
        encrypt_button = gr.Button("Step 4. Encrypt the image using FHE.")

        user_id = gr.Textbox(label="", max_lines=2, interactive=False, visible=False)

        # Display an image representation if possible
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2262
        encrypted_image = gr.Textbox(
            label="Encrypted image representation:", max_lines=2, interactive=False
        )

    gr.Markdown("## Server side")
    gr.Markdown(
        "The encrypted value is received by the server. The server can then compute the filter "
        "directly over encrypted values. Once the computation is finished, the server returns "
        "the encrypted results to the client."
    )

    with gr.Row():
        send_input_button = gr.Button("Step 5. Send the encrypted image to the server.")

        send_input_checkbox = gr.Checkbox(label="Encrypted image sent.", interactive=False)

    with gr.Row():
        execute_fhe_button = gr.Button("Step 6. Run FHE execution")

        fhe_execution_time = gr.Textbox(
            label="Total FHE execution time (in seconds).", max_lines=1, interactive=False
        )

    with gr.Row():
        get_output_button = gr.Button("Step 7. Receive the encrypted output image from the server.")

        # Display an image representation if possible
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2262
        encrypted_output_image = gr.Textbox(
            label="Encrypted output image representation:", max_lines=2, interactive=False
        )

    gr.Markdown("## Client side")
    gr.Markdown(
        "The encrypted output is sent back to client, who can finally decrypt it with its "
        "private key. Only the client is aware of the original image and its transformed version."
    )

    decrypt_button = gr.Button("Step 8. Decrypt the output")

    # Final input vs output display
    with gr.Row():
        original_image = gr.Image(
            input_image.value,
            label=f"Input image ({INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}):",
            interactive=False,
        )
        original_image.style(height=256, width=256)

        output_image = gr.Image(
            label=f"Output image ({INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}):", interactive=False
        )
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
        outputs=[original_image, encrypted_image],
    )

    # Button to send the encodings to the server using post method
    send_input_button.click(
        send_input, inputs=[user_id, image_filter], outputs=[send_input_checkbox]
    )

    # Button to send the encodings to the server using post method
    execute_fhe_button.click(run_fhe, inputs=[user_id, image_filter], outputs=[fhe_execution_time])

    # Button to send the encodings to the server using post method
    get_output_button.click(
        get_output, inputs=[user_id, image_filter], outputs=[encrypted_output_image]
    )

    # Button to decrypt the output on the client side
    decrypt_button.click(
        decrypt_output,
        inputs=[user_id, image_filter],
        outputs=[output_image, keygen_checkbox, send_input_checkbox],
    )

    gr.Markdown(
        "The app was built with [Concrete-ML](https://github.com/zama-ai/concrete-ml), a "
        "Privacy-Preserving Machine Learning (PPML) open-source set of tools by [Zama](https://zama.ai/). "
        "Try it yourself and don't forget to star on Github &#11088;."
    )

demo.launch(share=False)
