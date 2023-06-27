# Conventions

The only place we can say we/our is in a blog post written by a named author. Anything signed Zama should not use we/our. So, no "We" or "Our" in our documentation.

Let's use following conventions for the docs. If a new convention needs to be decided, let's agree and then add it here.

## Style

1. The documentation can address the user directly e.g., "you can find out more at ..", but this
   style should not be abused, do not be too colloquial !
1. The documentation should not refer to its writers or to Concrete ML developers/Zama as "we",
   and thus it should not use "our" either.

## Terms

1. Use hyphenated versions for portmanteaus, unless the term is in the dictionary, example:
   - bit-width \[not: bitwidth, bit width\]
   - input-set, data-set
   - database, codebase is fine (not data-base, code-base)
   - de-quantize/re-quantize
1. "an FHE program" not "a FHE program"
1. Machine Learning or machine learning, depends on the context
1. google is a verb ("you can google" but not "you can Google") : but try to avoid this
1. Programs:
   - Jupyter
   - Concrete ML (no Concrete-ML)
   - pytest except when title where it is capitalized
   - Python
   - torch (for the code) and PyTorch (for the product)
   - scikit, sklearn and scikit-learn are acceptable
   - Hummingbird
   - Docker (for the product) or docker (for the binary)
   - Poetry (for the product) or poetry (for the binary)
   - Make (for the product) or make (for the command line)
   - PoissonRegression or Poisson Regression (depends on the context, we'll fix it in Zama)
   - macOS
   - bare OS

## Variables

1. Variables names should use ticks: `variable_name` or `variable_name=10.5`
1. If a number is just stated, not in a code block, then it's just 10.5 not `10.5`.

## Titles

- Main titles (with a single #) are `Capitalized at Each Letter`
- Sub titles (with two or more #s) are `Capitalized only for first letter`

## Links

- Use links from doc root (i.e., ../../user/explanation/quantization.md) instead of using the smallest number of ../ that works; it makes files easier to move
