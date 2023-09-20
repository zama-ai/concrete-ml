# Hybrid model deployment

FHE enables cloud applications to process private user data without running the risk of data leaks. Furthermore, deploying ML models in the cloud is advantageous as it eases model updates, allows to scale to large numbers of users by using large amounts of compute power, and protects model IP by keeping the model on a trusted server instead of the client device.

However, not all applications can be easily converted to FHE computation and the computation cost of FHE may make a full conversion exceed latency requirements.

Hybrid models are a compromise between on-premise or on-device deployment and full cloud deployment. Hybrid deployment means parts of the model are executed on the client side and parts are executed in FHE on the server side. Concrete ML supports hybrid deployment of neural network models such as MLP, CNN and Large Language-Models.

{% hint style="warning" %}
If model IP protection is important, care must be taken in choosing the parts of a model to be executed on the cloud. Some
black-box model stealing attacks rely on knowledge distillation
or on differential methods. As a general rule, the difficulty
to steal a machine learning model is proportional to the size of the model, in terms of numbers of parameters and model depth.
{% endhint %}

The hybrid model deployment API provides an easy way to integrate the [standard deployment procedure](client_server.md) into neural network style models that are compiled with [`compile_brevitas_qat_model`](../developer-guide/api/concrete.ml.torch.compile.md#kbdfunctionkbd-compilebrevitasqatmodel) or [`compile_torch_model`](../developer-guide/api/concrete.ml.torch.compile.md#kbdfunctionkbd-compiletorchmodel).

## Compilation

To use hybrid model deployment, the first step is to define what part of the PyTorch neural network model must be executed in FHE. The model part must be a `nn.Module` and is identified by its key in the original model's `.named_modules()`.

```python
from torch import nn
from concrete.ml.torch.hybrid_model import HybridFHEModel

class FCSmall(nn.Module):
    """Torch model for the tests."""

    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.seq(x)

model = FCSmall(10)
model_name = "FCSmall"
submodule_name = "seq_0"

# Prints ['', 'seq', 'seq.0', 'seq.1', 'seq.2']
print([k for (k, _) in model.named_modules()])

# Create a hybrid model
hybrid_model = HybridFHEModel(model, [submodule_name])
hybrid_model.compile_model(
    inputs,
    n_bits=8,
)

models_dir = Path(__file__).parent / "compiled_models"
models_dir.mkdir(exist_ok=True)
model_dir = models_dir / model_name
hybrid_model.save_and_clear_private_info(model_dir, via_mlir=True)
```

## Server Side Deployment

<!--pytest-codeblocks:cont-->

The [`save_and_clear_private_info`](<>) function serializes the FHE circuits
corresponding to the various parts of the model that were chosen to be moved
server-side. Furthermore it saves all necessary information required
to serve these sub-models with FHE, using the [`FHEModelDev`](../developer-guide/api/concrete.ml.deployment.fhe_client_server.md#kbdclasskbd-fhemodeldev) class.

The [`FHEModelServer`](../developer-guide/api/concrete.ml.deployment.fhe_client_server.md#kbdclasskbd-fhemodelserver) class should be used to create a server application that creates end-points to serve these sub-models:

```
from concrete.ml.deployment import FHEModelServer
MODULES = { model_name: { submodule_name: {"path":  model_dir / "seq_0" }}}
return FHEModelServer(str(MODULES[model_name][submodule_name]["path"]))
```

For more information about serving FHE models, see the [client/server section](client_server.md#serving).

## Client Side

A client application that deploys a model with hybrid deployment can be developed
in a very similar manner to on-premise deployment: the model is loaded normally with Pytorch, but an extra step is required to specify the remote endpoint and the model parts that are to be executed remotely.

<!--pytest-codeblocks:cont-->

```python
# Modify model to use remote FHE server instead of local weights
hybrid_model = HybridFHEModel(
    model,
    submodule_name,
    server_remote_address="http://0.0.0.0:8000",
    model_name=f"{model_name}",
    verbose=False,
)
```

Next, the client application must obtain the parameters necessary to encrypt and
quantize data, as detailed in the [client/server documentation](client_server.md#production-deployment).

<!--pytest-codeblocks:cont-->

```
path_to_clients = Path(__file__).parent / "clients"
hybrid_model.init_client(path_to_clients=path_to_clients)
```

When the client application is ready to make inference requests to the server, it must
set the operation mode of the `HybridFHEModel` instance to `HybridFHEMode.REMOTE`:

<!--pytest-codeblocks:cont-->

```
for module in hybrid_model.remote_modules.values():
    module.fhe_local_mode = HybridFHEMode.REMOTE    
```

When performing inference with the `HybridFHEModel` instance, `hybrid_model`, only the regular `forward` method is called, as if the model was fully deployed locally:

<!--pytest-codeblocks:cont-->

```python
hybrid_model.forward(torch.randn((dim, )))
```

When calling `forward`, the `HybridFHEModel` handles, for each model part that is deployed remotely, all the necessary intermediate steps: quantizing the data, encrypting it, makes the request to the server using `requests` Python module, decrypting and de-quantizing the result.
