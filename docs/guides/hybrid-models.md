# Hybrid models

This document explains how to use Concrete ML API to deploy hybrid models in Fully Homomorphic Encryption (FHE).

## Introduction

FHE allows cloud applications to process private user data securely, minimizing the risk of data leaks. Deploying machine learning (ML) models in the cloud offers several advantages:

- Simplifies model updates.
- Scales to large user bases by leveraging substantial compute power.
- Protects model's Intellectual Property (IP) by keeping the model on a trusted server rather than on client devices.

However, not all applications can be easily converted to FHE computation. The high computation cost of FHE might exceed latency requirements for full conversion.

Hybrid models provide a balance between on-device deployment and cloud-based deployment. This approach involves:

- Executing parts of the model on the client side.
- Securely processing other parts with FHE on the server side.

Concrete ML supports hybrid deployment for various neural network models, including Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), and Large Language Models(LLM).

{% hint style="warning" %}
To protect model IP, carefully choose the model parts to execute in the cloud. Some black-box model stealing attacks use knowledge distillation or differential methods. Generally, the difficulty of stealing a machine learning model increases with the model's size, number of parameters, and depth.
{% endhint %}

The hybrid model deployment API simplifies integrating the [standard deployment procedure](client_server.md) into neural network style models that are compiled with [`compile_brevitas_qat_model`](../references/api/concrete.ml.torch.compile.md#function-compile_brevitas_qat_model) or [`compile_torch_model`](../references/api/concrete.ml.torch.compile.md#function-compile_torch_model).

## Compilation

To use hybrid model deployment, the first step is to define which part of the PyTorch neural network model must be executed in FHE. Ensure the model part is a `nn.Module` and is identified by its key in the original model's `.named_modules()`.

Here is an example:

```python
import numpy as np
import os
import torch

from pathlib import Path
from torch import nn

from concrete.ml.torch.hybrid_model import HybridFHEModel, tuple_to_underscore_str
from concrete.ml.deployment import FHEModelServer


class FCSmall(nn.Module):
    """Torch model for the tests."""

    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.seq(x)

dim = 10
model = FCSmall(dim)
model_name = "FCSmall"
submodule_name = "seq.0"

inputs = torch.Tensor(np.random.uniform(size=(10, dim)))
# Prints ['', 'seq', 'seq.0', 'seq.1', 'seq.2']
print([k for (k, _) in model.named_modules()])

# Create a hybrid model
hybrid_model = HybridFHEModel(model, [submodule_name])
hybrid_model.compile_model(
    inputs,
    n_bits=8,
)

models_dir = Path(os.path.abspath('')) / "compiled_models"
models_dir.mkdir(exist_ok=True)
model_dir = models_dir / model_name

```

<!--pytest-codeblocks:skip-->

```python
hybrid_model.save_and_clear_private_info(model_dir, via_mlir=True)
```

## Server Side Deployment

The [`save_and_clear_private_info`](../references/api/concrete.ml.torch.hybrid_model.md#method-save_and_clear_private_info) functions as follows:

- Serializes the FHE circuits for the model parts chosen to be server-side.
- Saves the client-side model, removing the weights of the layers transferred to the server.
- Saves all necessary information required to serve these sub-models with FHE using the [`FHEModelDev`](../references/api/concrete.ml.deployment.fhe_client_server.md#class-fhemodeldev) class.

To create a server application that serves these sub-models, use the [`FHEModelServer`](../references/api/concrete.ml.deployment.fhe_client_server.md#class-fhemodelserver) class:

<!--pytest-codeblocks:skip-->

```
input_shape_subdir = tuple_to_underscore_str( (1,) + inputs.shape[1:] )
MODULES = { model_name: { submodule_name: {"path":  model_dir / submodule_name / input_shape_subdir }}}
server =  FHEModelServer(str(MODULES[model_name][submodule_name]["path"]))
```

For more information about serving FHE models, see the [client/server section](client_server.md#serving).

## Client Side

You can develop a client application that deploys a model with hybrid deployment in a very similar manner to on-premise deployment: Use PyTorch to load the model normally, but specify the remote endpoint and the part of the model to be executed remotely.

<!--pytest-codeblocks:skip-->

```python
# Modify model to use remote FHE server instead of local weights
hybrid_model = HybridFHEModel(
    model,  # PyTorch or Brevitas model
    submodule_name,
    server_remote_address="http://0.0.0.0:8000",
    model_name=f"{model_name}",
    verbose=False,
)
```

Next, obtain the parameters necessary to encrypt and quantize data, as detailed in the [client/server documentation](client_server.md#production-deployment).

<!--pytest-codeblocks:skip-->

```python
path_to_clients = Path(__file__).parent / "clients"
hybrid_model.init_client(path_to_clients=path_to_clients)
```

When the client application is ready to make inference requests to the server, set the operation mode of the `HybridFHEModel` instance to `HybridFHEMode.REMOTE`:

<!--pytest-codeblocks:skip-->

```python
for module in hybrid_model.remote_modules.values():
    module.fhe_local_mode = HybridFHEMode.REMOTE    
```

For inference with the `HybridFHEModel` instance, `hybrid_model`, call the regular `forward` method as if the model was fully deployed locally::

<!--pytest-codeblocks:skip-->

```python
hybrid_model(torch.randn((dim, )))
```

When calling `HybridFHEModel`, it handles all the necessary intermediate steps for each model part deployed remotely, including:

- Quantizing the data.
- Encrypting the data.
- Making the request to the server using `requests` Python module.
- Decrypting and de-quantizing the result.
