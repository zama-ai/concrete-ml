---
name: Internal RC Version Release
about: Template to prepare an internal release candidate used for internal testing
title: "Release vX.Y.Z-rcN"
---
<!-- Make sure to set the proper version in the issue template -->

# Description

This checklist shows the steps to make an internal release candidate (RC). This release candidate will
tag the `main` branch so that we can later identify the commit corresponding to the RC. Then
we will be able to install the RC in a virtual environment to use in testing or benchmarking. 

Please follow the steps below:

## 1. Preliminary steps:

- [ ] Checkout latest `main`
- [ ] Find the current version configured in the source tree: look in `src/concrete/ml/version.py`. Find the current RC tag in the tag list in the repo: [https://github.com/zama-ai/concrete-ml-internal/tags](https://github.com/zama-ai/concrete-ml-internal/tags)

## 2. Update version if necessary 

You can **skip the following steps** if the version in `version.py` is already newer than the ones in the tag list

- [ ] Choose a new version number: 
  - The new version number should be in the format `X.Y.Z-rcN` where `X.Y.Z` is the current version number and `N` is the new release candidate number
- [ ] Create a new branch, such as `git checkout -b chore/rc_release_X.Y.Z-rcN`. Update the project version to the **new** version with the following command line:

```bash
VERSION=X.Y.Z-rc? make set_version
```
- [ ] After doing `make set_version` in the previous step, you need to push your new branch, make a PR and get it approved. It will then be merged to `main` branch

## 3. Tag the new RC release

- [ ] Checkout latest `main`
- [ ] Ensure you have GPG keys, by running `gpg --list-secret-keys --keyid-format=long`. The list should not be empty. If it is empty, follow the instructions on [how to associate a GPG key with your email](https://docs.github.com/en/authentication/managing-commit-signature-verification/associating-an-email-with-your-gpg-key). Make sure you **remember the passphrase**
- [ ] Run `make release`. You may be prompted for the passphrase (you can save it with a password manager)

## 4. Check that the RC was made successfully

- [ ] Go to [https://github.com/zama-ai/concrete-ml-internal/tags](https://github.com/zama-ai/concrete-ml-internal/tags). You should see your new RC version in the tags
- [ ] Go to [Github Actions](https://github.com/zama-ai/concrete-ml-internal/actions) and wait for the CI workflow to complete for your new tag. You can filter using the Branch menu to select only your new tag. The CI should show a green check :heavy_check_mark: 

## 5. Optional: prepare the future RC version number
- [ ] Repeat **Step 2** above to set the `main` branch to a new, future, RC version for a future release and avoid doing step 2 at the next rc release
