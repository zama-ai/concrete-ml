# Documenting

## Using Sphinx

One can simply create docs with Sphinx and open them, by doing:

```shell
make docs
```

Reminder that this needs to be done in docker.

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

conveniently builds and open the doc at the end.
