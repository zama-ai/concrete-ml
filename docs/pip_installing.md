# Installing with pip

## Conditions on OS

**Concrete-ML** can be run on **Linux based OSes** as well as **macOS on x86 CPUs** (for **Apple Silicon CPUs**, there is currently no support, Docker needs to be used). For **Windows**, one can:

- either use **WSL on Windows**, which is a Linux based OS: **Concrete-ML** will work as long as the package is not installed in the /mnt/c/ directory, which corresponds to the host OS filesystem.
- or use Docker, see [instructions here](docker_installing.md)

These hardware requirements are dictated by **Concrete-Lib**.

## Actual installation steps

To install **Concrete-ML** from PyPi, run the following:

```shell
pip install concrete-ml
```

```{note}
Note that **concrete-ml** installs **concrete-numpy** with all extras, including `pygraphviz` to draw graphs.
```

```{WARNING}
`pygraphviz` requires `graphviz` packages being installed on your OS, see <a href="https://pygraphviz.github.io/documentation/stable/install.html">https://pygraphviz.github.io/documentation/stable/install.html</a>
```

```{DANGER}
`graphviz` packages are binary packages that won't automatically be installed by pip.
Do check <a href="https://pygraphviz.github.io/documentation/stable/install.html">https://pygraphviz.github.io/documentation/stable/install.html</a> for instructions on how to install `graphviz` for `pygraphviz`.
```
