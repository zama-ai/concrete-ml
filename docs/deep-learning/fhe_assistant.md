# Debugging Models

This section provides a set of tools and guidelines to help users build optimized FHE-compatible models.

## Virtual library

The _Virtual Lib_ in Concrete-ML is a prototype that provides drop-in replacements for Concrete-Numpy's compiler, allowing users to simulate FHE execution, including any probabilistic behavior FHE may induce. The Virtual Library comes from Concrete-Numpy, where it is called [Virtual Circuits](https://docs.zama.ai/concrete-numpy/tutorials/virtual_circuits).

The Virtual Lib can be useful when developing and iterating on an ML model implementation. For example, you can check that your model is compatible in terms of operands (all integers) with the Virtual Lib compilation. Then, you can check how many bits your ML model would require, which can give you hints about ways it could be modified to compile it to an actual FHE Circuit. As FHE non-linear models work with integers up to 16 bits, with a tradeoff between number of bits and FHE execution speed, the Virtual Lib can help to find the optimal model design.

The Virtual Lib, being pure Python and not requiring crypto key generation, can be much faster than the actual compilation and FHE execution, thus allowing for faster iterations, debugging and FHE simulation, regardless of the bit-width used. For example, this was used for the red/blue contours in the [Classifier Comparison notebook](../built-in-models/ml_examples.md), as computing in FHE for the whole grid and all the classifiers would take significant time.

The following example shows how to use the Virtual Lib in Concrete-ML. Simply add `use_virtual_lib = True` and `enable_unsafe_features = True` in a `Configuration`. The result of the compilation will then be a simulated circuit that allows for more precision or simulated FHE execution.

```python
from sklearn.datasets import fetch_openml, make_circles
from concrete.ml.sklearn import RandomForestClassifier
from concrete.numpy import Configuration

debug_config = Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location="~/.cml_keycache",
    p_error=None,
    global_p_error=None,
)

n_bits = 2
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.6, random_state=0)
concrete_clf = RandomForestClassifier(
    n_bits=n_bits, n_estimators=10, max_depth=5
)
concrete_clf.fit(X, y)

concrete_clf.compile(X, debug_config, use_virtual_lib=True)

y_preds_clear = concrete_clf.predict(X)
```

## Compilation debugging

The following example produces a neural network that is not FHE-compatible:

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

Upon execution, the compiler will raise the following error within the graph representation:

```
Function you are trying to compile cannot be converted to MLIR:

%0 = _onnx__Gemm_0                    # EncryptedTensor<int7, shape=(1, 2)>        ∈ [-64, 63]
%1 = [[ 33 -27  ...   22 -29]]        # ClearTensor<int7, shape=(2, 120)>          ∈ [-63, 62]
%2 = matmul(%0, %1)                   # EncryptedTensor<int14, shape=(1, 120)>     ∈ [-4973, 4828]
%3 = subgraph(%2)                     # EncryptedTensor<uint7, shape=(1, 120)>     ∈ [0, 126]
%4 = [[ 16   6  ...   10  54]]        # ClearTensor<int7, shape=(120, 120)>        ∈ [-63, 63]
%5 = matmul(%3, %4)                   # EncryptedTensor<int17, shape=(1, 120)>     ∈ [-45632, 43208]
%6 = subgraph(%5)                     # EncryptedTensor<uint7, shape=(1, 120)>     ∈ [0, 126]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ table lookups are only supported on circuits with up to 16-bit integers
%7 = [[ -7 -52] ... [-12  62]]        # ClearTensor<int7, shape=(120, 2)>          ∈ [-63, 62]
%8 = matmul(%6, %7)                   # EncryptedTensor<int16, shape=(1, 2)>       ∈ [-26971, 29843]
return %8
```

Knowing that a linear/dense layer is implemented as a matrix multiplication, it can determine which parts of the op-graph listing in the exception message above correspond to which layers.

Layer weights initialization:

```
%1 = [[ 33 -27  ...   22 -29]]        # ClearTensor<int7, shape=(2, 120)>         
%4 = [[ 16   6  ...   10  54]]        # ClearTensor<int7, shape=(120, 120)>   
%7 = [[ -7 -52] ... [-12  62]]        # ClearTensor<int7, shape=(120, 2)> 
```

Input data:

```
%0 = _onnx__Gemm_0                    # EncryptedTensor<int7, shape=(1, 2)>   
```

First dense layer and activation function:

```
%2 = matmul(%0, %1)                   # EncryptedTensor<int14, shape=(1, 120)>    
%3 = subgraph(%2)                     # EncryptedTensor<uint7, shape=(1, 120)>        
```

Second dense layer and activation function:

```
%5 = matmul(%3, %4)                   # EncryptedTensor<int17, shape=(1, 120)>    
%6 = subgraph(%5)                     # EncryptedTensor<uint7, shape=(1, 120)>  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ table lookups are only supported on circuits with up to 16-bit integers
```

Third dense layer:

```
%8 = matmul(%6, %7)                   # EncryptedTensor<int16, shape=(1, 2)> 
return %8
```

We can see here that the error is in the second layer because the product has exceeded the 16-bit precision limit. This error is only detected when the PBS operations are actually applied.

However, reducing the number of neurons in this layer resolves the error and makes the network FHE-compatible:

<!--pytest-codeblocks:cont-->

```python
torch_model = SimpleNet(10)

quantized_numpy_module = compile_torch_model(
    torch_model,
    torch_input,
    n_bits=7,
)
```

## Complexity analysis

In FHE, univariate functions are encoded as table lookups, which are then implemented using Programmable Bootstrapping (PBS). PBS is a powerful technique but will require significantly more computing resources, and thus time, than simpler encrypted operations such as matrix multiplications, convolution or additions.

Furthermore, the cost of PBS will depend on the bit-width of the compiled circuit. Every additional bit in the maximum bit-width raises the complexity of the PBS by a significant factor. It may be of interest to the model developer, then, to determine the bit-width of the circuit and the amount of PBS it performs.

This can be done by inspecting the MLIR code produced by the compiler:

### Concrete-ML Model

<!--pytest-codeblocks:cont-->

```python
torch_model = SimpleNet(10)

quantized_numpy_module = compile_torch_model(
    torch_model,
    torch_input,
    n_bits=7,
    show_mlir=True,
)
```

### Compiled MLIR model

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

Getting the bit-width of the circuit is then simply:

<!--pytest-codeblocks:cont-->

```python
print(quantized_numpy_module.forward_fhe.graph.maximum_integer_bit_width())
```

Decreasing the number of bits and the number of PBS induces large reductions in the computation time of the compiled circuit.
