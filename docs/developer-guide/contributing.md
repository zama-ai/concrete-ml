# Contributing

There are three ways to contribute to Concrete ML:

- You can open issues to report bugs and typos and to suggest ideas.
- You can become an official contributor but you need to sign our Contributor License Agreement (CLA) on your first contribution. Our CLA-bot will guide you through the process when you will open a Pull Request on Github.
- You can also provide new tutorials or use-cases, showing what can be done with the library. The more examples we have, the better and clearer it is for the other users.

## 1. Creating a new branch

To create your branch, you have to use the issue ID somewhere in the branch name:

```shell
git checkout -b {feat|fix|refactor|test|benchmark|doc|style|chore}/short-description_$issue_id
git checkout -b short-description_$issue_id
git checkout -b $issue_id_short-description
```

For example:

```shell
git checkout -b feat/explicit-tlu_11
git checkout -b tracing_indexing_42
git checkout -b 42_tracing_indexing
```

## 2. Before committing

### 2.1 Conformance

Each commit to Concrete ML should conform to the standards of the project. You can let the development tools fix some issues automatically with the following command:

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

Concrete ML uses a consistent commit naming scheme, and you are expected to follow it as well (the CI will make sure you do). The accepted format can be printed to your terminal by running:

```shell
make show_scope
```

For example:

```shell
git commit -m "feat: implement bounds checking"
git commit -m "feat(debugging): add an helper function to draw intermediate representation"
git commit -m "fix(tracing): fix a bug that crashed PyTorch tracer"
```

Just a reminder that commit messages are checked in the conformance step and are rejected if they don't follow the rules. To learn more about conventional commits, check [this](https://www.conventionalcommits.org/en/v1.0.0/) page.

## 4. Rebasing

You should rebase on top of the `main` branch before you create your pull request. Merge commits are not allowed, so rebasing on `main` before pushing gives you the best chance of to avoid rewriting parts of your PR later if conflicts arise with other PRs being merged. After you commit changes to your new branch, you can use the following commands to rebase:

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
