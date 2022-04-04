# Concrete Stack

**Concrete-ML** is built on top of Zama’s **Concrete** stack. It uses [**Concrete-Numpy**](https://github.com/zama-ai/concrete-numpy), which itself uses the [**Concrete-Compiler**](https://pypi.org/project/concrete-compiler/).

The **Concrete-Compiler** takes MLIR code as input representing a computation circuit and compiles it to an executable using **Concrete** primitives to perform the computations.

We refer the reader to **Concrete-Numpy** [documentation](https://docs.zama.ai/concrete-numpy/stable/) and, more generally, to the documentation of the whole **Concrete-Framework** for [more information](https://docs.zama.ai).
