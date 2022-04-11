---
name: Dot Release
about: Issue template to prepare a dot release step by step.
title: "Release vX.Y.Z"
---
<!-- Make sure to set the proper version in the issue template -->
Please check all steps if it was either done/already done, at the end of a release all check boxes must have been checked.

Release check-list:
<!-- Note that some of these steps will be automated in the future -->
If it was not already done:
- [ ] Choose the version number, e.g. `vX.Y.Z` following semantic versioning: https://semver.org/
- [ ] Update the project version to `X.Y.Z` by running:

```bash
VERSION=X.Y.Z make set_version
```

In the rest of this issue, **0.1.x is given as an example**, but of course, it should be replaced by the proper release branch number (0.2.x, 1.0.x etc).

- [ ] Be sure to have pushed the new version to `release/0.1.x` branch


Then:
- [ ] Checkout the commit for release
- [ ] Call `make release`, which creates a signed tag (requires GPG keys setup, see [here](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification)) and pushes it
- [ ] Wait for the release workflow ([here](https://github.com/zama-ai/concrete-ml-internal/actions)) to finish and check everything went well. It should take a bit longer than a classical build

For public releases:
- [ ] Check you have the public remote in your repo with the command below
```bash
git remote -vv
# Should output something similar to this
origin  git@github.com:zama-ai/concrete-ml-internal.git (fetch)
origin  git@github.com:zama-ai/concrete-ml-internal.git (push)
public  git@github.com:zama-ai/concrete-ml.git (fetch)
public  git@github.com:zama-ai/concrete-ml.git (push)
```
- [ ] If you don't have the public remote add it with the command below
```bash
# This is for the ssh remote
git remote add public git@github.com:zama-ai/concrete-ml.git
# For the https remote
git remote add public https://github.com/zama-ai/concrete-ml.git
# And update the remotes
git remote update
```

- [ ] Checkout the commit for release
- [ ] Push the release commit to the public repo on the main branch
```bash
git push public HEAD:release/0.1.x
```
- [ ] Download all assets from the private GitHub release but `concrete-ml-internal-x.y.z.tar.gz` and `concrete-ml-internal-x.y.z.zip`. Releases are found [here](https://github.com/zama-ai/concrete-ml-internal/releases)
- [ ] Update the text for the private release to include a human readable summary
- [ ] Fine tune the changelog, notably to not contain links to issues (which are sometimes added by squash-and-merge), and remove commit messages which are a bit useless (e.g., "fix review")
- [ ] Change `concrete-ml-internal` references to `concrete-ml` in the downloaded CHANGELOG.md
- [ ] Create a GitHub release on the public repo [here](https://github.com/zama-ai/concrete-ml/releases/new) by selecting the tag you just pushed. The name of the release is `vX.Y.Z`, exactly as the name of the tag.
- [ ] Copy the text from the private release to the public release DO NOT VALIDATE THE RELEASE YET
- [ ] Change `concrete-ml-internal` references to `concrete-ml` in the release text
- [ ] For the docker image replace `ghcr.io/zama-ai` by `zamafhe`
- [ ] Upload the assets from the private release to the public release, except `concrete-ml-internal-x.y.z.tar.gz` and `concrete-ml-internal-x.y.z.zip` that you have normally not downloaded and which will be added automatically by GitHub

- [ ] Validate the public release

At the end, to avoid any confusion
- [ ] it may be a good idea to remove the public repo:

```bash
git remote remove public
````

and check

```bash
git remote -vv
# Should output something similar to this
origin  git@github.com:zama-ai/concrete-ml-internal.git (fetch)
origin  git@github.com:zama-ai/concrete-ml-internal.git (push)
```

All done!
