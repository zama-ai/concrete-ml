# Documentation

## Using GitBook

Documentation with GitBook is done mainly by pushing content on GitHub. GitBook then pulls the docs from the repository, and publishes.. In most cases, GitBook is just a reflexion of what is available in GitHub.

There are however some use-cases where we want to modify documentation directly in GitBook (and then, push the modifications to GitHub), for example when the documentation is modified by a person outside of our organization. In this case, a GitHub branch is created, and a GitHub space is associated to it: modifications are done in this space, and automatically pushed to the branch. Once the modifications are done, one can simply create a pull-request, to finally merge modifications on the main branch.

## Using Sphinx

Documenation can alternatively be built using Sphinx:

```shell
make docs
```

The documentation contains both files written by hand by developers (the .md files) and files automatically created by parsing the source files.

Then to open it go to `docs/_build/html/index.html` or use the follwing command:

```shell
make open_docs
```

To build and open the docs at the same time, use:

```shell
make docs_and_open_docs
```


