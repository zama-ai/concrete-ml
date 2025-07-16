
# 🧪 Use Case – Hybrid Fine-Tuning of LLaMA-3.2-1B with LoRA and Remote deployment


This use case explores how to fine-tune the LLaMA-3.2-1B model on the Orca Math Word Problems dataset using LoRA (Low-Rank Adaptation) in a hybrid privacy-preserving setup.

This setup builds on the HybridModel, which is designed to isolate sensitive computations locally while offloading encrypted workloads remotely.

- The client side handles the dataset and the execution of non-linear layers locally, without exposing sensitive inputs.
- The server side processes the compute-intensive linear layers, which are executed under Fully Homomorphic Encryption (FHE) to ensure privacy even during remote computation.

The training and inference pipelines are designed so that:

- Only encrypted representations are sent to the server,
- Raw data never leaves the client,
- The bulk of the model’s compute is securely offloaded, reducing the burden on local resources.


## ⚙️ Getting Started

If you're using an AWS instance (or any fresh server), you can configure it automatically by running:

```sudo ./machine_configuration.sh```

This script:

- Creates a `.venv` virtual environment,
- Installs all necessary dependencies listed in `requirements_server.txt`.


## 🌐 Remote Execution Setup

After configuring the machine, update the `server_remote_address` in your script or notebook with the public IP of your server. Then, you can run either of the following to start the fine-tuning process:

- `dev.py` — script version
- `HybridFineTuningOfLLaMa.ipynb` — notebook version

## 📊 Benchmark
