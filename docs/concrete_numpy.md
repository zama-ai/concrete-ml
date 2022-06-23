# Concrete-Numpy Summation

{% hint style='info' %}
FIXME: to be done, Benoit
{% endhint %}

# Pre and post processing

In theory, it is possible to combine **Concrete-Numpy** with **Concrete-ML** such that the server can apply some pre or post processing before or after the execution of the model on the data. However this brings some complexity has all operations must be done in the quantized realm.

So currently there is no support for pre and post processing in FHE. Data must arrive to the FHE model already pre-processed and post-processing (if there is any) has to be done on the client machine.

We might add support for this in the future.
