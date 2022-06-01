# Installing with pip

**Concrete-ML** can be run on **Linux based OSes** as well as **macOS on x86 CPUs**. These hardware requirements are dictated by **Concrete-Lib**.

Do note that since **WSL on Windows** is a Linux based OS, **Concrete-ML** will work as long as the package is not mandated in the /mnt/c/ directory, which corresponds to the host OS filesystem.

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
