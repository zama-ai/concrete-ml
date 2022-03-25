# Concrete ML

Concrete ML is an open-source set of tools which aims to simplify the use of fully homomorphic encryption (FHE) for data scientists. Particular care was given to the simplicity of our python package, in order to make it usable by any data scientist, without any prior cryptography knowledge. Notably, our APIs are as close as possible from scikit-learn and torch APIs, to simplify adoption by our users.

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Links](#links)
- [For end users](#for-end-users)
  - [Installation](#installation)
  - [A simple ML example with a scikit-learn comparison](#a-simple-ml-example-with-a-scikit-learn-comparison)
  - [A simple DL example with a torch comparison](#a-simple-dl-example-with-a-torch-comparison)
- [For developers](#for-developers)
  - [Project setup](#project-setup)
  - [Documenting](#documenting)
  - [Developing](#developing)
  - [Contributing](#contributing)
- [License](#license)

<!-- mdformat-toc end -->

## Links

- [documentation](https://docs.zama.ai/concrete-ml/main/)
- [community website](https://community.zama.ai/c/concrete-ml)
- [demos](https://docs.zama.ai/concrete-ml/main/user/advanced_examples/index.html)

## For end users

### Installation

The preferred way to use Concrete ML is through docker. You can get our docker image by pulling the latest docker image:

`docker pull zamafhe/concrete-ml:latest`

To install Concrete ML from PyPi, run the following:

`pip install concrete-ml`

You can find more detailed installation instructions in [installing.md](docs/user/basics/installing.md)

### A simple ML example with a scikit-learn comparison

Let's show by example how simple it is to mimic the use of scikit-learn models with Concrete ML.

```python
print("FIXME (Benoit): to be added from https://docs.preprod.zama.ai/concrete-ml/main/user/howto/simple_example_sklearn.html")
```

### A simple DL example with a torch comparison

Let's show by example how simple it is to mimic the use of torch models with Concrete ML.

```python
print("FIXME (Benoit): to be added from https://docs.preprod.zama.ai/concrete-ml/main/user/howto/simple_example_torch.html")
```

## For developers

### Project setup

Installation steps are described in [project_setup.md](docs/dev/howto/project_setup.md).
Information about how to use Docker for development are available in [docker.md](docs/dev/howto/docker.md).

### Documenting

Some information about how to build the documentation of Concrete ML are [available](docs/dev/howto/documenting.md). Notably, our documentation is pushed to [https://docs.zama.ai/concrete-ml/](https://docs.zama.ai/concrete-ml/).

### Developing

Some information about the infrastructure of Concrete ML and some of the core elements we use are available [here](docs/dev/explanation/). Notably, an in-depth look at what is done in Concrete ML is available in [onnx_use_for_compilation.md](docs/dev/explanation/onnx_use_for_compilation.md).

### Contributing

Information about how to contribute are available in [contributing.md](docs/dev/howto/contributing.md).

## License

This software is distributed under the BSD-3-Clause-Clear license. If you have any questions, please contact us at hello@zama.ai.
