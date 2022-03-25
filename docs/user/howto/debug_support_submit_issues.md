# Debug / Get Support / Submit Issues

This version of **Concrete ML** is a first version of the product, meaning that it is not completely finished, contains several bugs (would they be known or unknown at this time), and will improve over time with feedback from early users.

Here are some ways to debug your problems. If nothing seems conclusive, you can still report the issue, as explained in a later section of this page.

## Is it a bug by the framework or by the user?

First, we encourage the user to have a look at:

- the error message received
- the documentation of the product
- the known limits of the product.

Once you have tried to see if the bug was not your own, it is time to go further.

## Is the inputset sufficiently representative?

A bug may happen if ever the inputset, which is internally used by the compilation core to set bit widths of some intermediate data, is not sufficiently representative. If ever, with all the inputs in the inputset, it appears that an intermediate data can be represented an `n`-bit integer, but for a particular computation, this same intermediate data needs a bit more bits to be represented, the FHE execution for this computation will result in a wrong output (as typically in integer overflows in classical programs).

So, in general, when a bug appears, it may be a good idea to enlarge the inputset, and try to have random-looking inputs in this latter, following distribution of inputs used with the function.

## Having a reproducible bug

Once you're sure it is a bug, it would be nice to try to:

- make it highly reproducible: e.g., by reducing as much the randomness as possible; e.g., if you can find an input which fails, there is no reason to let the input random
- reduce it to the smallest possible bug: it is easier to investigate bugs which are small, so when you have an issue, please try to reduce to a smaller issue, notably with less lines of code, smaller parameters, less complex function to compile, faster scripts etc.

## Asking the community

You can directly ask the developers and community about your issue on our Discourse server (link on the right of the top menu).

Hopefully, it is just a misunderstanding or a small mistake on your side, that we can help you fix easily. And, the good point with your feedback is that, once we have heard the problem or misunderstanding, we can make the documentation even clearer (by adding to the FAQ).

## Submitting an issue

To simplify our work and let us reproduce your bug easily, we need all the information we can get. So, in addition to your python script, the following information would be very useful:

- reproducibility rate you see on your side
- any insight you might have on the bug
- any workaround you have been able to find

Remember, **Concrete ML** is a project where we are open to contributions, more information at [Contributing](../../dev/howto/contributing.md).

In case you have a bug, which is reproducible, that you have reduced to a small piece of code,  we have our issue tracker (link on the right of the top menu). Remember that a well-described short issue is an issue which is more likely to be studied and fixed. The more issues we receive, the better the product will be.
