# Upgrading Concrete ML In Your Project

This guide aims to help developers who are upgrading from older versions of Concrete ML. Although we cannot cover everything, we have compiled the most important points to assist you in your transition. If you encounter any issues, please do not hesitate to reach out to https://community.zama.ai or ask on the fhe.org Discord.

## Upgrading to 1.0.0

Please take note of the following changes when upgrading to version 1.0.0:

- `execute_in_fhe` argument in `.predict()` methods has been replaced by `fhe="disable|simulate|execute"`. The `disable` option runs the model in Python, while `simulate` performs an FHE simulation, and `execute` provides actual FHE execution.

- `encrypt_run_decrypt` function can now __only__ be executed in FHE. For FHE simulations, please use the `simulate` function instead.

- In some models, the `forward_fhe` function has been renamed to `fhe_circuit` for consistency across all models.

- `verbose_compilation` parameter has been renamed to `verbose` in the compile functions.

- `compilation_artifacts` parameter has been renamed to `artifacts`.

- `use_virtual_lib` parameter in `concrete.ml.torch` has been removed from the following functions:

  - `compile_onnx_model`
  - `compile_torch_model`
  - `compile_brevitas_qat_model`
    This means that models are now always converted to FHE, and only FHE-friendly models can be compiled.
