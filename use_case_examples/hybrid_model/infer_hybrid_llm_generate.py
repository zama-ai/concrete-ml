"""Showcase for the hybrid model converter."""
import json
import time
from pathlib import Path

import torch
from torch.backends import mps
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from concrete.ml.torch.hybrid_model import HybridFHEMode, HybridFHEModel

if __name__ == "__main__":
    # Use configuration dumped by compilation
    with open("configuration.json", "r") as file:
        configuration = json.load(file)
    module_names = configuration["module_names"]

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Get GPT2 from Hugging Face
    model_name = "gpt2"
    # Avoid having / in the string
    model_name_no_special_char = model_name.replace("/", "_")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
    )

    # Modify model to use remote FHE server instead of local weights
    hybrid_model = HybridFHEModel(
        model,
        module_names,
        server_remote_address="http://0.0.0.0:8000",
        model_name=f"{model_name}",
        verbose=False,
    )
    path_to_clients = Path(__file__).parent / "clients"
    hybrid_model.init_client(path_to_clients=path_to_clients)
    hybrid_model.set_fhe_mode(HybridFHEMode.REMOTE)

    # Run example
    while True:
        # Take inputs
        num_tokens = input("Number of tokens:\n").strip()
        if not num_tokens:
            num_tokens = 5
        else:
            num_tokens = int(num_tokens)
        prompt = input("Prompt:\n")
        if not prompt:
            prompt = "Computations on encrypted data can help"

        # Encode and send to device
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        assert isinstance(input_ids, torch.Tensor)
        input_ids = input_ids.to(device=device)

        print("*" * 10)
        print("*" * 10)
        print(f"{input_ids.shape[1]} tokens in '{prompt}'")
        print("*" * 10)
        print("*" * 10)

        # Print words as they are generated
        streamer = TextStreamer(tokenizer=tokenizer)
        start = time.time()
        output_ids = model.generate(
            input_ids, max_new_tokens=num_tokens, use_cache=True, streamer=streamer
        )
        end = time.time()
        generated = tokenizer.decode(output_ids[0])

        print(f"{end - start} seconds to generate")
        print("*" * 10)
        print("*" * 10)
        print(generated)
        print("*" * 10)
        print("*" * 10)
