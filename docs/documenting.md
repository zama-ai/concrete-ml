# Document

Primary, our documentation is available within `GitBook`. Additionally, we provide a `Sphinx` documentation.

## Using GitBook

Documentation with GitBook is done mainly by pushing content on GitHub. Then, GitBook finds information in our repository, and publish it to this [main](https://app.gitbook.com/o/-MIF05xPVoj0l_wnOGB7/s/MkXyATSxN2odyoLKjl8J/) link. Thus, in most cases, GitBook is just a _reflexion_ of what is available in GitHub.

There are however some use-cases where we want to _modify_ documentation directly in GitBook (and then, push the modifications to GitHub): for example, when the documentation is modified by a person outside of our organization. In this case, a GitHub branch is created, and a GitHub space is associated to it: modifications are done in this space, and automatically pushed to the branch. Once the modifications are done, one can simply create a pull-request, to finally merge modifications in our main.

## Using Sphinx

One can simply create docs with Sphinx and open them, by doing:

```shell
make docs
```

The documentation contains both files written by hand by developers (the .md files) and files automatically created by parsing the source files.

### Opening doc

```shell
make open_docs
```

Or simply open `docs/_build/html/index.html`.

Remark that a

```shell
make docs_and_open_docs
```

conveniently builds and opens the doc at the end.
