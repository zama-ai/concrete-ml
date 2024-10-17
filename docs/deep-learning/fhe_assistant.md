# Debugging models

This section provides a set of tools and guidelines to help users debug errors and build optimized models that are compatible with Fully Homomorphic Encryption (FHE).

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

## Common compilation errors

#### 1. TLU input maximum bit-width is exceeded

**Error message**: `this [N]-bit value is used as an input to a table lookup`

**Cause**: This error can occur when `rounding_threshold_bits` is not used and accumulated intermediate values in the computation exceed 16 bits. To pinpoint the model layer that causes the error, Concrete ML provides the [bitwidth_and_range_report](../references/api/concrete.ml.quantization.quantized_module.md#method-bitwidth_and_range_report) helper function. To use this function, the model must be compiled first so that it can be [simulated](fhe_assistant.md#simulation).

**Possible solutions**:

- Reduce quantization `n_bits`. However, this may reduce accuracy. When quantization `n_bits` must be below 6, it is best to use [Quantization Aware Training](../deep-learning/fhe_friendly_models.md).
- Use `rounding_threshold_bits`. This feature is described [here](../explanations/advanced_features.md#rounded-activations-and-quantizers). It is recommended to use the [`fhe.Exactness.APPROXIMATE`](../references/api/concrete.ml.torch.compile.md#function-compile_torch_model) setting, and set the rounding bits to 1 or 2 bits higher than the quantization `n_bits`
- Use [pruning](../explanations/pruning.md)

#### 2. No crypto-parameters can be found

**Error message**: `RuntimeError: NoParametersFound`

**Cause**: This error occurs when cryptosystem parameters can not be found for the model bit-width, rounding mode and requested `p_error`, when using `rounding_threshold_bits` in the `compile_torch_model` function. With `rounding_threshold_bits` set, the 16-bit accumulator limit is relaxed, so the `this [N]-bit value is used as an input to a table lookup` does not occur. However, cryptosystem-parameters may still not exist for the model to be compiled.

**Possible solutions**: The solutions in this case are similar to the ones for the previous error: reducing bit-width, or reducing the `rounding_threshold_bits`, or using the [`fhe.Exactness.APPROXIMATE`](../references/api/concrete.ml.torch.compile.md#function-compile_torch_model) rounding method can help. Additionally adjusting the tolerance for one-off errors using the `p_error` parameter can help, as explained in [this section](../explanations/advanced_features.md#approximate-computations).

#### 3. Quantization import failed

**Error message**: `Error occurred during quantization aware training (QAT) import [...] Are you missing a QuantIdentity layer in your Brevitas model?`.

**Cause**: This error occurs when the model imported as a quantized-aware training model lacks quantization operators. See [this guide](../deep-learning/fhe_friendly_models.md) on how to use Brevitas layers. This error message indicates that some layers do not take inputs quantized through `QuantIdentity` layers.

A common example is related to the concatenation operator. Suppose two tensors `x` and `y` are produced by two layers and need to be concatenated:

<!--pytest-codeblocks:skip-->

```python
x = self.dense1(x)
y = self.dense2(y)
z = torch.cat([x, y])
```

In the example above, the `x` and `y` layers need quantization before being concatenated.

**Possible solutions**:

1. If the error occurs for the first layer of the model: Add a  `QuantIdentity` layer in your model and apply it on the input of the `forward` function, before the first layer is computed.
1. If the error occurs for a concatenation or addition layer: Add a new `QuantIdentity` layer in your model. Suppose it is called `quant_concat`. In the `forward` function, before concatenation of `x` and `y`, apply it to both tensors that are concatenated. The usage of a common `Quantidentity` layer to quantize both tensors that are concatenated ensures that they have the same scale:

<!--pytest-codeblocks:skip-->

```python
z = torch.cat([self.quant_concat(x), self.quant_concat(y)])
```

## PBS complexity and optimization

In FHE, univariate functions are encoded as Table Lookups, which are then implemented using [Programmable Bootstrapping (PBS)](../getting-started/concepts.md#cryptography-concepts). PBS is a powerful technique but requires significantly more computing resources compared to simpler encrypted operations such as matrix multiplications, convolution, or additions.

Furthermore, the cost of PBS depends on the bit-width of the compiled circuit. Every additional bit in the maximum bit-width significantly increase the complexity of the PBS. Therefore, it is important to determine the bit-width of the circuit and the amount of PBS it performs in order to optimize the performance.

To inspect the MLIR code produced by the compiler, use the following command:

<!--pytest-codeblocks:skip-->

```python
print(quantized_numpy_module.fhe_circuit.mlir)
```

Example output:

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

In the MLIR code, there are several calls to `FHELinalg.apply_mapped_lookup_table` and `FHELinalg.apply_lookup_table`. These calls apply PBS to the cells of their input tensors. For example, in the code above, the inputs are: `tensor<1x5x!FHE.eint<15>>` for both the first and last `apply_mapped_lookup_table` call. Thus, the PBS is applied 10 times, corresponding to the size of every encrypted tensor, which is 1x5 multiplied by 2.

To retrieve the bit-width of the circuit, use this command:

<!--pytest-codeblocks:skip-->

```python
print(quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width())
```

Reducing the number of bits and the number of PBS applications can significantly decrease the computation time of the compiled circuit.
