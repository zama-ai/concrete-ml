# Contributing

There are three ways to contribute to Concrete ML:

- You can open issues to report bugs and typos and to suggest ideas.
- You can become an official contributor but you need to sign our Contributor License Agreement (CLA) on your first contribution. Our CLA-bot will guide you through the process when you will open a Pull Request on Github.
- You can also provide new tutorials or use-cases, showing what can be done with the library. The more examples we have, the better and clearer it is for the other users.

## 1. Setting up the project

First, you need to [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) the [Concrete ML](../README.md) repository and properly set up the project by following the steps provided [here](project_setup.md).

## 2. Creating a new branch

When creating your branch, make sure the name follows the expected format :

```shell
git checkout -b {feat|fix|docs|chore}/short_description_$(issue_id)
git checkout -b {feat|fix|docs|chore}/$(issue_id)_short_description
```

For example:

```shell
git checkout -b feat/add_avgpool_operator_470
git checkout -b feat/470_add_avgpool_operator
```

## 3. Before committing

### 3.1 Conformance

Each commit to Concrete ML should conform to the standards of the project. You can let the development tools fix some issues automatically with the following command:

```shell
make conformance
```

Additionally, you will need to make sure that the following command does not return any error (`pcc`: pre-commit checks):

```shell
make pcc
```

### 3.2 Testing

Your code must be well documented, provide extensive tests if any feature has been added and must not break other tests.
To execute all tests, please run the following command. Be aware that running all tests can take up to an hour.

```shell
make pytest
```

You need to make sure you get 100% code coverage. The `make pytest` command checks that by default and will fail with a coverage report at the end should some lines of your code not be executed during testing.

If your coverage is below 100%, you should write more tests and then create the pull request. If you ignore this warning and create the PR, checks will fail and your PR will not be merged.

There may be cases where covering your code is not possible (an exception that cannot be triggered in normal execution circumstances). In those cases, you may be allowed to disable coverage for some specific lines. This should be the exception rather than the rule, and reviewers will ask why some lines are not covered. If it appears they can be covered, then the PR won't be accepted in that state.

## 4. Committing

Concrete ML uses a consistent commit naming scheme and you are expected to follow it as well. The accepted format can be printed to your terminal by running:

```shell
make show_commit_rules
```

For example:

```shell
git commit -m "feat: support AVGPool2d operator"
git commit -m "fix: fix AVGPool2d operator"
```

Just a reminder that commit messages are checked in the conformance step and are rejected if they don't follow the rules. To learn more about conventional commits, check [this page](https://www.conventionalcommits.org/en/v1.0.0/).

## 5. Rebasing

You should rebase on top of the repository's `main` branch before you create your pull request. Merge commits are not allowed, so rebasing on `main` before pushing gives you the best chance of to avoid rewriting parts of your PR later if conflicts arise with other PRs being merged. After you commit changes to your forked repository, you can use the following commands to rebase your main branch with Concrete ML's one:

```shell
# Add the Concrete ML repository as remote, named "upstream" 
git remote add upstream git@github.com:zama-ai/concrete-ml.git

# Fetch all last branches and changes from Concrete ML
git fetch upstream

# Checkout to your local main branch
git checkout main

# Rebase on top of main
git rebase upstream/main

# If there are conflicts during the rebase, resolve them
# and continue the rebase with the following command
git rebase --continue

# Push the latest version of your local main to your remote forked repository
git push --force origin main
```

You can learn more about rebasing [here](https://git-scm.com/docs/git-rebase).

## 6. Open a pull-request

You can now open a pull-request [in the Concrete ML repository](https://github.com/zama-ai/concrete-ml/pulls). For more details on how to do so from a forked repository, please read GitHub's [official documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) on the subject.
