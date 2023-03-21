# Upgrading Concrete ML In Your Project

This document is an help for developers who update from older versions of Concrete-ML. It is difficult to be exhaustive, but we try to list the most important points to help you in your transition. And in any case, if troubles, don't hesitate to reach out https://community.zama.ai or ask on fhe.org discord.

## Upgrading to 1.0.0

`encrypt_run_decrypt` is now restricted to execution in FHE. For simulation with the Virtual Library, one uses `simulate` function

`forward_fhe` has been renamed `fhe_circuit` in some models, such that it is now `fhe_circuit` for all models.

`verbose_compilation` has been renamed `verbose`.

`compilation_artifacts` has been renamed `artifacts`.

`execute_in_fhe` argument in `.predict()` methods has been replaced by `fhe = "disable|simulate|execute"` where disable runs the model in python, simulate is a FHE simulation and execute provides the actual FHE execution.
