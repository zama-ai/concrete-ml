# Debugging models

This section provides a set of tools and guidelines to help users build optimized FHE-compatible models. It discusses FHE simulation, the key-cache functionality that helps speed-up FHE result debugging, and gives a guide to evaluate circuit complexity.

## Simulation

The [simulation functionality](../explanations/compilation.md#fhe-simulation) of Concrete ML provides a way to evaluate, using clear data, the results that ML models produce on encrypted data. The simulation includes any probabilistic behavior FHE may induce. The simulation is implemented with [Concrete's simulation](https://docs.zama.ai/concrete/execution-analysis/simulation).

The simulation mode can be useful when developing and iterating on an ML model implementation. As FHE non-linear models work with integers up to 16 bits, with a trade-off between the number of bits and the FHE execution speed, the simulation can help to find the optimal model design.

Simulation is much faster than FHE execution. This allows for faster debugging and model optimization. For example, this was used for the red/blue contours in the [Classifier Comparison notebook](../tutorials/ml_examples.md), as computing in FHE for the whole grid and all the classifiers would take significant time.

The following example shows how to use the simulation mode in Concrete ML.

```python
from sklearn.datasets import fetch_openml, make_circles
from concrete.ml.sklearn import RandomForestClassifier

n_bits = 2
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.6, random_state=0)
concrete_clf = RandomForestClassifier(
    n_bits=n_bits, n_estimators=10, max_depth=5
)
concrete_clf.fit(X, y)

concrete_clf.compile(X)

# Running the model using FHE-simulation
y_preds_clear = concrete_clf.predict(X, fhe="simulate")
```

## Caching keys during debugging

It is possible to avoid re-generating the keys of the models you are debugging. This feature is unsafe and should not be used in production. Here is an example that shows how to enable key-caching:

```python
from sklearn.datasets import fetch_openml, make_circles
from concrete.ml.sklearn import RandomForestClassifier
from concrete.fhe import Configuration
debug_config = Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache",
)

n_bits = 2
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.6, random_state=0)
concrete_clf = RandomForestClassifier(
    n_bits=n_bits, n_estimators=10, max_depth=5
)
concrete_clf.fit(X, y)

concrete_clf.compile(X, debug_config)
```

## Compilation error debugging

Compilation errors that signal that the ML model is not FHE compatible are usually of two types:

1. TLU input maximum bit-width is exceeded
1. No crypto-parameters can be found for the ML model: `RuntimeError: NoParametersFound` is raised by the compiler

The following produces a neural network that is not FHE-compatible:

```python
import numpy
import torch

from torch import nn
from concrete.ml.torch.compile import compile_torch_model

N_FEAT = 2
class SimpleNet(nn.Module):
    """Simple MLP with PyTorch"""

    def __init__(self, n_hidden=30):
        super().__init__()
        self.fc1 = nn.Linear(in_features=N_FEAT, out_features=n_hidden)
        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.fc3 = nn.Linear(in_features=n_hidden, out_features=2)


    def forward(self, x):
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


torch_input = torch.randn(100, N_FEAT)
torch_model = SimpleNet(120)
try:
    quantized_numpy_module = compile_torch_model(
        torch_model,
        torch_input,
        n_bits=7,
    )
except RuntimeError as err:
    print(err)
```

Upon execution, the Compiler will raise the following error within the graph representation:

```
Function you are trying to compile cannot be compiled:

%0 = _x                               # EncryptedTensor<int7, shape=(1, 2)>           ∈ [-64, 63]
%1 = [[ -9  18  ...   30  34]]        # ClearTensor<int7, shape=(2, 120)>             ∈ [-62, 63]              @ /fc1/Gemm.matmul
%2 = matmul(%0, %1)                   # EncryptedTensor<int14, shape=(1, 120)>        ∈ [-5834, 5770]          @ /fc1/Gemm.matmul
%3 = subgraph(%2)                     # EncryptedTensor<uint7, shape=(1, 120)>        ∈ [0, 127]
%4 = [[-36   6  ...   27 -11]]        # ClearTensor<int7, shape=(120, 120)>           ∈ [-63, 63]              @ /fc2/Gemm.matmul
%5 = matmul(%3, %4)                   # EncryptedTensor<int17, shape=(1, 120)>        ∈ [-34666, 37702]        @ /fc2/Gemm.matmul
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this 17-bit value is used as an input to a table lookup
```

The error `this 17-bit value is used as an input to a table lookup` indicates that the 16-bit limit on the input of the Table Lookup (TLU) operation has been exceeded. To pinpoint the model layer that causes the error, Concrete ML provides the [bitwidth_and_range_report](../references/api/concrete.ml.quantization.quantized_module.md#method-bitwidth_and_range_report) helper function. First, the model must be compiled so that it can be [simulated](fhe_assistant.md#simulation).

### Fixing compilation errors

To make this network FHE-compatible one can apply several techniques:

1. use [rounded accumulators](../explanations/advanced_features.md#rounded-activations-and-quantizers) by specifying the `rounding_threshold_bits` parameter. Please evaluate the accuracy of the model using simulation if you use this feature, as it may impact accuracy. Setting a value 2-bit higher than the quantization `n_bits` should be a good start.

<!--pytest-codeblocks:cont-->

```python
torch_model = SimpleNet(20)

quantized_numpy_module = compile_torch_model(
    torch_model,
    torch_input,
    n_bits=6,
    rounding_threshold_bits=7,
)
```

2. reduce the accumulator bit-width of the second layer named `fc2`. To do this, a simple solution is to reduce the number of neurons, as it is proportional to the bit-width.

<!--pytest-codeblocks:cont-->

```python
torch_model = SimpleNet(10)

quantized_numpy_module = compile_torch_model(
    torch_model,
    torch_input,
    n_bits=7,
)
```

3. adjust the tolerance for one-off errors using the `p_error` parameter. See [this section for more explanation](../explanations/advanced_features.md#approximate-computations) on this tolerance.

<!--pytest-codeblocks:cont-->

```python
torch_model = SimpleNet(10)

quantized_numpy_module = compile_torch_model(
    torch_model,
    torch_input,
    n_bits=7,
    p_error=0.01
)
```

## Complexity analysis

In FHE, univariate functions are encoded as table lookups, which are then implemented using Programmable Bootstrapping (PBS). PBS is a powerful technique but will require significantly more computing resources, and thus time, compared to simpler encrypted operations such as matrix multiplications, convolution, or additions.

Furthermore, the cost of PBS will depend on the bit-width of the compiled circuit. Every additional bit in the maximum bit-width raises the complexity of the PBS by a significant factor. It may be of interest to the model developer, then, to determine the bit-width of the circuit and the amount of PBS it performs.

This can be done by inspecting the MLIR code produced by the Compiler:

<!--pytest-codeblocks:cont-->

```python
print(quantized_numpy_module.fhe_circuit.mlir)
```

```
MLIR
--------------------------------------------------------------------------------
module {
  func.func @main(%arg0: tensor<1x2x!FHE.eint<15>>) -> tensor<1x2x!FHE.eint<15>> {
    %cst = arith.constant dense<16384> : tensor<1xi16>
    %0 = "FHELinalg.sub_eint_int"(%arg0, %cst) : (tensor<1x2x!FHE.eint<15>>, tensor<1xi16>) -> tensor<1x2x!FHE.eint<15>>
    %cst_0 = arith.constant dense<[[-13, 43], [-31, 63], [1, -44], [-61, 20], [31, 2]]> : tensor<5x2xi16>
    %cst_1 = arith.constant dense<[[-45, 57, 19, 50, -63], [32, 37, 2, 52, -60], [-41, 25, -1, 31, -26], [-51, -40, -53, 0, 4], [20, -25, 56, 54, -23]]> : tensor<5x5xi16>
    %cst_2 = arith.constant dense<[[-56, -50, 57, 37, -22], [14, -1, 57, -63, 3]]> : tensor<2x5xi16>
    %c16384_i16 = arith.constant 16384 : i16
    %1 = "FHELinalg.matmul_eint_int"(%0, %cst_2) : (tensor<1x2x!FHE.eint<15>>, tensor<2x5xi16>) -> tensor<1x5x!FHE.eint<15>>
    %cst_3 = tensor.from_elements %c16384_i16 : tensor<1xi16>
    %cst_4 = tensor.from_elements %c16384_i16 : tensor<1xi16>
    %2 = "FHELinalg.add_eint_int"(%1, %cst_4) : (tensor<1x5x!FHE.eint<15>>, tensor<1xi16>) -> tensor<1x5x!FHE.eint<15>>
    %cst_5 = arith.constant

: tensor<5x32768xi64>
    %cst_6 = arith.constant dense<[[0, 1, 2, 3, 4]]> : tensor<1x5xindex>
    %3 = "FHELinalg.apply_mapped_lookup_table"(%2, %cst_5, %cst_6) : (tensor<1x5x!FHE.eint<15>>, tensor<5x32768xi64>, tensor<1x5xindex>) -> tensor<1x5x!FHE.eint<15>>
    %4 = "FHELinalg.matmul_eint_int"(%3, %cst_1) : (tensor<1x5x!FHE.eint<15>>, tensor<5x5xi16>) -> tensor<1x5x!FHE.eint<15>>
    %5 = "FHELinalg.add_eint_int"(%4, %cst_3) : (tensor<1x5x!FHE.eint<15>>, tensor<1xi16>) -> tensor<1x5x!FHE.eint<15>>
    %cst_7 = arith.constant

: tensor<5x32768xi64>
    %6 = "FHELinalg.apply_mapped_lookup_table"(%5, %cst_7, %cst_6) : (tensor<1x5x!FHE.eint<15>>, tensor<5x32768xi64>, tensor<1x5xindex>) -> tensor<1x5x!FHE.eint<15>>
    %7 = "FHELinalg.matmul_eint_int"(%6, %cst_0) : (tensor<1x5x!FHE.eint<15>>, tensor<5x2xi16>) -> tensor<1x2x!FHE.eint<15>>
    return %7 : tensor<1x2x!FHE.eint<15>>

  }
}
--------------------------------------------------------------------------------
```

There are several calls to `FHELinalg.apply_mapped_lookup_table` and `FHELinalg.apply_lookup_table`. These calls apply PBS to the cells of their input tensors. Their inputs in the listing above are: `tensor<1x2x!FHE.eint<8>>` for the first and last call and `tensor<1x50x!FHE.eint<8>>` for the two calls in the middle. Thus, PBS is applied 104 times.

Retrieving the bit-width of the circuit is then simply:

<!--pytest-codeblocks:cont-->

```python
print(quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width())
```

Decreasing the number of bits and the number of PBS applications induces large reductions in the computation time of the compiled circuit.
