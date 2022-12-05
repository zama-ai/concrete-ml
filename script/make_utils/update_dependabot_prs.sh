#!/usr/bin/env bash

# A script that pull each branch that starts by "dependabot/" does a git commit --amend and force push the branch to origin  (to update the PR)

# This script is useful when you have a lot of PRs that need to be updated

# Usage: ./update-dependabot-prs.sh

# Note: This script assumes that you have a remote called "origin"


# Get all branches that start with "dependabot/" from origin
branches=$(git ls-remote origin | grep "dependabot/" | awk '{print $2}' | sed 's/refs\/heads\///')

# Remember which branch we are currently in
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Loop through each branch
for branch in $branches; do
  # Checkout the branch
  git checkout "$branch"

  # Amend the commit
  git commit --amend --no-edit

  # Force push the branch to origin
  git push origin "$branch" --force

  # Checkout main
  git checkout main

  # Delete branch locally
  git branch -D "$branch"
done

# Checkout the branch we were in before
git checkout "$current_branch"
