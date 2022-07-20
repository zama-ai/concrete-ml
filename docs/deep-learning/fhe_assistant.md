# Debugging Models

This section provides a set of tools and guidelines to help users build optimized FHE compatible models.

## Virtual Lib

The Virtual Lib in Concrete-ML is a prototype that provides drop-in replacements for Concrete-Numpy's compiler that allow users to simulate what would happen when converting a model to FHE without the current bit width constraint, as well as quickly simulating the behavior with 8 bits or less without actually doing the FHE computations.

The Virtual Lib can be useful when developing and iterating on an ML model implementation. For example, you can check that your model is compatible in terms of operands (all integers) with the Virtual Lib compilation. Then, you can check how many bits your ML model would require, which can give you hints as to how it should be modified if you want to compile it to an actual FHE Circuit (not a simulated one) that only supports 8 bits of integer precision.

The Virtual Lib, being pure Python and not requiring crypto key generation, can be much faster than the actual compilation and FHE execution, thus allowing for faster iterations, debugging and FHE simulation, regardless of the bit width used. This was for example used for the red/blue contours in the [Classifier Comparison notebook](../built-in-models/ml_examples.md), as computing in FHE for the whole grid and all the classifiers would take significant time.

The following example shows how to use the Virtual Lib in Concrete-ML. Simply add `use_virtual_lib = True` and `enable_unsafe_features = True` in a `Configuration`. The result of the compilation will then be a simulated circuit that allows for more precision or simulated FHE execution.

```python
from sklearn.datasets import fetch_openml, make_circles
from concrete.ml.sklearn import RandomForestClassifier
from concrete.numpy import Configuration
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

concrete_clf.compile(X, debug_config, use_virtual_lib=True)

y_preds_clear = concrete_clf.predict(X)
```

## Compilation debugging

The following example produces a neural network that is not FHE compatible:

```python
import numpy
import torch

from torch import nn
from concrete.ml.torch.compile import compile_torch_model

N_FEAT = 2
class SimpleNet(nn.Module):
    """Simple MLP with torch"""

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
        n_bits = 3,
    )
except RuntimeError as err:
    print(err)
```

Upon execution, the compiler will raise the following error:

```
%0 = [[-1 -3] [ ... ] [-2  2]]        # ClearTensor<int3, shape=(120, 2)>
 %1 = [[ 1  3 -2 ...  1  2  0]]        # ClearTensor<int3, shape=(120, 120)>
 %2 = [[ 2  0  3 ... -2 -2 -1]]        # ClearTensor<int3, shape=(2, 120)>
 %3 = _onnx__Gemm_0                    # EncryptedTensor<uint5, shape=(1, 2)>
 %4 = -15                              # ClearScalar<int5>
 %5 = add(%3, %4)                      # EncryptedTensor<int6, shape=(1, 2)>
 %6 = subgraph(%5)                     # EncryptedTensor<int3, shape=(1, 2)>
 %7 = matmul(%6, %2)                   # EncryptedTensor<int6, shape=(1, 120)>
 %8 = subgraph(%7)                     # EncryptedTensor<uint3, shape=(1, 120)>
 %9 = matmul(%8, %1)                   # EncryptedTensor<int9, shape=(1, 120)>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only up to 8-bit integers are supported
%10 = subgraph(%9)                     # EncryptedTensor<uint3, shape=(1, 120)>
%11 = matmul(%10, %0)                  # EncryptedTensor<int8, shape=(1, 2)>
%12 = subgraph(%11)                    # EncryptedTensor<uint5, shape=(1, 2)>
return %12
```

Knowing that a linear/dense layer is implemented as a matrix multiplication, it can determined which parts of the op-graph listing in the exception message above correspond to which layers:

Layer weights initialization:

```
%0 = [[-1 -3] [ ... ] [-2  2]]        # ClearTensor<int3, shape=(120, 2)>
 %1 = [[ 1  3 -2 ...  1  2  0]]        # ClearTensor<int3, shape=(120, 120)>
 %2 = [[ 2  0  3 ... -2 -2 -1]]        # ClearTensor<int3, shape=(2, 120)>
```

Input processing and quantization:

```
 %3 = _onnx__Gemm_0                    # EncryptedTensor<uint5, shape=(1, 2)>
 %4 = -15                              # ClearScalar<int5>
 %5 = add(%3, %4)                      # EncryptedTensor<int6, shape=(1, 2)>
 %6 = subgraph(%5)                     # EncryptedTensor<int3, shape=(1, 2)>
```

First dense layer and activation function:

```
%7 = matmul(%6, %2)                   # EncryptedTensor<int6, shape=(1, 120)>
%8 = subgraph(%7)                     # EncryptedTensor<uint3, shape=(1, 120)>
```

Second dense layer and activation function:

```
%9 = matmul(%8, %1)                   # EncryptedTensor<int9, shape=(1, 120)>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only up to 8-bit integers are supported
%10 = subgraph(%9)                     # EncryptedTensor<uint3, shape=(1, 120)>
```

Third dense layer and output quantization:

```
%11 = matmul(%10, %0)                  # EncryptedTensor<int8, shape=(1, 2)>
%12 = subgraph(%11)                    # EncryptedTensor<uint5, shape=(1, 2)>
return %12
```

We can see here that the error is in the second layer. Reducing the number of neurons in this layer will resolve the error and make the network FHE compatible:

<!--pytest-codeblocks:cont-->

```python
torch_model = SimpleNet(50)
try:
    quantized_numpy_module = compile_torch_model(
        torch_model,
        torch_input,
        n_bits = 3,
    )
except RuntimeError as err:
    print(err)
```

## Complexity analysis

In FHE, univariate functions are encoded as table lookups, which are then implemented using Programmable Bootstrapping (PBS). Programmable bootstrapping is a powerful technique, but will require significantly more compute resources and thus time than more simpler encrypted operations such matrix multiplications, convolution or additions.

Furthermore, the cost of a PBS will depend on the bitwidth of the compiled circuit. Every additional bit in the maximum bitwidth raises the complexity of the PBS by a significant factor. It thus may be of interest to the model developer to determine the bitwidth of the circuit and the number of PBS it performs.

This can be done by inspecting the MLIR code produced by the compiler:

### Concrete-ML Model

<!--pytest-codeblocks:cont-->

```python
torch_model = SimpleNet(50)
try:
    quantized_numpy_module = compile_torch_model(
        torch_model,
        torch_input,
        n_bits = 3,
        show_mlir=True,
    )
except RuntimeError as err:
    print(err)
```

### Compiled MLIR model

```
%cst = arith.constant dense<...> : tensor<50x2xi9>
%cst_0 = arith.constant dense<...>
%cst_1 = arith.constant dense<...> : tensor<2x50xi9>
%c-14_i9 = arith.constant -14 : i9
%c128_i9 = arith.constant 128 : i9
%c128_i9_2 = arith.constant 128 : i9
%c128_i9_3 = arith.constant 128 : i9
%c128_i9_4 = arith.constant 128 : i9
%hack_0_c-14_i9 = tensor.from_elements %c-14_i9 : tensor<1xi9>
%0 = "FHELinalg.add_eint_int"(%arg0, %hack_0_c-14_i9) : (tensor<1x2x!FHE.eint<8>>, tensor<1xi9>) -> tensor<1x2x!FHE.eint<8>>
%hack_1_c128_i9_4 = tensor.from_elements %c128_i9_4 : tensor<1xi9>
%1 = "FHELinalg.add_eint_int"(%0, %hack_1_c128_i9_4) : (tensor<1x2x!FHE.eint<8>>, tensor<1xi9>) -> tensor<1x2x!FHE.eint<8>>
%cst_5 = arith.constant dense<...> : tensor<256xi64>
%2 = "FHELinalg.apply_lookup_table"(%1, %cst_5) : (tensor<1x2x!FHE.eint<8>>, tensor<256xi64>) -> tensor<1x2x!FHE.eint<8>>

%3 = "FHELinalg.matmul_eint_int"(%2, %cst_1) : (tensor<1x2x!FHE.eint<8>>, tensor<2x50xi9>) -> tensor<1x50x!FHE.eint<8>>
%hack_4_c128_i9_3 = tensor.from_elements %c128_i9_3 : tensor<1xi9>
%4 = "FHELinalg.add_eint_int"(%3, %hack_4_c128_i9_3) : (tensor<1x50x!FHE.eint<8>>, tensor<1xi9>) -> tensor<1x50x!FHE.eint<8>>
%cst_6 = arith.constant dense<...> : tensor<34x256xi64>
%cst_7 = arith.constant dense<...]> : tensor<1x50xindex>
%5 = "FHELinalg.apply_mapped_lookup_table"(%4, %cst_6, %cst_7) : (tensor<1x50x!FHE.eint<8>>, tensor<34x256xi64>, tensor<1x50xindex>) -> tensor<1x50x!FHE.eint<8>>

%6 = "FHELinalg.matmul_eint_int"(%5, %cst_0) : (tensor<1x50x!FHE.eint<8>>, tensor<50x50xi9>) -> tensor<1x50x!FHE.eint<8>>
%hack_7_c128_i9_2 = tensor.from_elements %c128_i9_2 : tensor<1xi9>
%7 = "FHELinalg.add_eint_int"(%6, %hack_7_c128_i9_2) : (tensor<1x50x!FHE.eint<8>>, tensor<1xi9>) -> tensor<1x50x!FHE.eint<8>>
%cst_8 = arith.constant dense<...> : tensor<34x256xi64>
%cst_9 = arith.constant dense<...> : tensor<1x50xindex>
%8 = "FHELinalg.apply_mapped_lookup_table"(%7, %cst_8, %cst_9) : (tensor<1x50x!FHE.eint<8>>, tensor<34x256xi64>, tensor<1x50xindex>) -> tensor<1x50x!FHE.eint<8>>

%9 = "FHELinalg.matmul_eint_int"(%8, %cst) : (tensor<1x50x!FHE.eint<8>>, tensor<50x2xi9>) -> tensor<1x2x!FHE.eint<8>>
%hack_10_c128_i9 = tensor.from_elements %c128_i9 : tensor<1xi9>
%10 = "FHELinalg.add_eint_int"(%9, %hack_10_c128_i9) : (tensor<1x2x!FHE.eint<8>>, tensor<1xi9>) -> tensor<1x2x!FHE.eint<8>>
%cst_10 = arith.constant dense<...> : tensor<2x256xi64>
%cst_11 = arith.constant dense<[[0, 1]]> : tensor<1x2xindex>
%11 = "FHELinalg.apply_mapped_lookup_table"(%10, %cst_10, %cst_11) : (tensor<1x2x!FHE.eint<8>>, tensor<2x256xi64>, tensor<1x2xindex>) -> tensor<1x2x!FHE.eint<8>>
return %11 : tensor<1x2x!FHE.eint<8>>
```

We notice that we have calls to `FHELinalg.apply_mapped_lookup_table` and `FHELinalg.apply_lookup_table`. These calls apply PBS to the cells of their input tensors. Their inputs in the listing above are: `tensor<1x2x!FHE.eint<8>>` for the first and last call and `tensor<1x50x!FHE.eint<8>>` for the two calls in the middle. Thus PBS is applied 104 times.

Getting the bitwidth of the circuit is then simply:

<!--pytest-codeblocks:cont-->

```python
print(quantized_numpy_module.forward_fhe.graph.maximum_integer_bit_width())
```

Decreasing the number of bits and the number of PBS induces large reductions in the computation time of the compiled circuit.
