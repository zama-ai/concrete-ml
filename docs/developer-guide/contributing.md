# Contributing

There are three ways to contribute to Concrete-ML:

- You can open issues to report bugs and typos and to suggest ideas.
- You can ask to become an official contributor by emailing [hello@zama.ai](mailto:hello@zama.ai). Only approved contributors can send pull requests (PR), so please make sure to get in touch before you do.
- You can also provide new tutorials or use-cases, showing what can be done with the library. The more examples we have, the better and clearer it is for the other users.

## 1. Creating a new branch

Concrete-ML uses a consistent branch naming scheme, and you are expected to follow it as well. Here is the format, along with some examples:

```shell
git checkout -b {feat|fix|refactor|test|benchmark|doc|style|chore}/short-description_$issue_id
```

e.g.

```shell
git checkout -b feat/explicit-tlu_11
git checkout -b fix/tracing_indexing_42
```

## 2. Before committing

### 2.1 Conformance

Each commit to Concrete-ML should conform to the standards of the project. You can let the development tools fix some issues automatically with the following command:

```shell
make conformance
```

Conformance can be checked using the following command:

```shell
make pcc
```

### 2.2 Testing

Your code must be well documented, containing tests and not breaking other tests:

```shell
make pytest
```

You need to make sure you get 100% code coverage. The `make pytest` command checks that by default and will fail with a coverage report at the end should some lines of your code not be executed during testing.

If your coverage is below 100%, you should write more tests and then create the pull request. If you ignore this warning and create the PR, GitHub actions will fail and your PR will not be merged.

There may be cases where covering your code is not possible (an exception that cannot be triggered in normal execution circumstances). In those cases, you may be allowed to disable coverage for some specific lines. This should be the exception rather than the rule, and reviewers will ask why some lines are not covered. If it appears they can be covered, then the PR won't be accepted in that state.

## 3. Committing

Concrete-ML uses a consistent commit naming scheme, and you are expected to follow it as well (the CI will make sure you do). The accepted format can be printed to your terminal by running:

```shell
make show_scope
```

e.g.

```shell
git commit -m "feat: implement bounds checking"
git commit -m "feat(debugging): add an helper function to draw intermediate representation"
git commit -m "fix(tracing): fix a bug that crashed PyTorch tracer"
```

To learn more about conventional commits, check [this](https://www.conventionalcommits.org/en/v1.0.0/) page. Just a reminder that commit messages are checked in the comformance step and are rejected if they don't follow the rules.

## 4. Rebasing

You should rebase on top of the `main` branch before you create your pull request. Merge commits are not allowed, so rebasing on `main` before pushing gives you the best chance of avoiding having to rewrite parts of your PR later if conflicts arise with other PRs being merged. After you commit your changes to your new branch, you can use the following commands to rebase:

```shell
# fetch the list of active remote branches
git fetch --all --prune

# checkout to main
git checkout main

# pull the latest changes to main (--ff-only is there to prevent accidental commits to main)
git pull --ff-only

# checkout back to your branch
git checkout $YOUR_BRANCH

# rebase on top of main branch
git rebase main

# If there are conflicts during the rebase, resolve them
# and continue the rebase with the following command
git rebase --continue

# push the latest version of the local branch to remote
git push --force
```

You can learn more about rebasing [here](https://git-scm.com/docs/git-rebase).

## 5. Releases

Before any final release, Concrete-ML contributors go through a release candidate (RC) cycle. The idea is that once the codebase and documentations look ready for a release, you create an RC release by opening an issue with the release template [here](https://github.com/zama-ai/concrete-ml/issues/new?assignees=&labels=&template=release.md), starting with version `vX.Y.Zrc1` and then with versions `vX.Y.Zrc2`, `vX.Y.Zrc3`...

Once the last RC is deemed ready, open an issue with the release template using the last RC version from which you remove the `rc?` part (i.e. `v12.67.19` if your last RC version was `v12.67.19-rc4`) on [github](https://github.com/zama-ai/concrete-ml/issues/new?assignees=&labels=&template=release.md).
