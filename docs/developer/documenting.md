# Documentation

## Using GitBook

Documentation with GitBook is done mainly by pushing content on GitHub. GitBook then pulls the docs from the repository and publishes. In most cases, GitBook is just a mirror of what is available in GitHub.

There are, however, some use-cases where documentation can be modified directly in GitBook (and, then, push the modifications to GitHub), for example when the documentation is modified by a person outside of Zama. In this case, a GitHub branch is created, and a GitHub space is associated to it: modifications are done in this space and automatically pushed to the branch. Once the modifications have been completed, one can simply create a pull-request, to finally merge modifications on the main branch.
