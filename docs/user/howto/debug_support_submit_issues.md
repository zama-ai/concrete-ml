# Debug / Get Support / Submit Issues

This version of **Concrete ML** is a first version of the product, meaning that it is not completely finished, contains several bugs (both known and unknown at this time), and will improve over time with feedback from early users.

Here are some ways to debug your problems. If nothing seems conclusive, you can still report the issue, as explained in a later section of this page.

## Is it a bug by the framework or by the user?

First, we encourage the user to have a look at:

- the error message received
- the documentation of the product
- the known limits of the product

Once you have tried to see if the bug was not your own, it is time to go further.

## Is the inputset sufficiently representative?

A bug may happen if ever the inputset, which is internally used by the compilation core to set bitwidths of some intermediate data, is not sufficiently representative. With all the inputs in the inputset, it appears that intermediate data can be represented as an `n`-bit integer, but for a particular computation, this same intermediate data needs additional bits to be represented. The FHE execution for this computation will result in an incorrect output (as typically occurs in integer overflows in classical programs).

So, in general, when a bug appears, it may be a good idea to enlarge the inputset and try to have random-looking inputs in the latter, following distribution of inputs used with the function.

## Having a reproducible bug

Once you're sure it is a bug, it would be nice to try to:

- make it highly reproducible by reducing as much the randomness as possible. If you can find an input which fails, there is no reason to let the input remain random
- reduce it to the smallest possible bug. It is easier to investigate bugs which are small, so when you have an issue, please try to reduce to a smaller issue, notably with fewer lines of code, smaller parameters, less complex functions to compile, and faster scripts, etc.

## Asking the community

You can directly ask the developers and community about your issue on our Discourse server (link on the right of the top menu).

Hopefully, it is just a misunderstanding or a small mistake on your side that we can help you fix easily. Additionally, your feedback helps us make the documentation even clearer (by adding to the FAQ, for example).

## Submitting an issue

To simplify our work and let us reproduce your bug easily, we need all the information we can get. So, in addition to your python script, the following information is useful:

- the reproducibility rate you see on your side
- any insight you might have on the bug
- any workaround you have been able to find

Remember, **Concrete ML** is a project where we are open to contributions. You can find more information at [Contributing](../../dev/howto/contributing.md).

In case you have a reproducible bug that you have reduced to a small piece of code, we have our issue tracker in place (link on the right of the top menu). Remember that a well-described short issue is an issue which is more likely to be studied and fixed. The more issues we receive, the better the product will be.
