# Virtual Lib

## What is the Virtual Lib?

The Virtual Lib in **Concrete-ML** is a prototype that provides drop-in replacements for **Concrete-Numpy**, Compiler and Circuit that allow users to simulate what would happen when converting a model to FHE without the current bit width constraint, or to more quickly simulate the behavior with 8 bits or less as there are no FHE computations.

In other words, you can use the compile functions from the **Concrete-ML** package by passing `use_virtual_lib = True` and using a `Configuration` with `enable_unsafe_features = True`. You will then get a simulated circuit that allows you to use more than the current 8 bits of precision allowed by the **Concrete** stack. It is also a faster way to measure the potential FHE accuracy with 8 bits or less. It is something we used for the red/blue contours in the [Classifier Comparison notebook](../../user/advanced_examples/ClassifierComparison.ipynb), as computing in FHE for the whole grid and all the classifiers would be very long.

## What should it be used for?

The Virtual Lib can be useful when developing and iterating on an ML model implementation. For example, you can check that your model is compatible in terms of operands (all integers) with the Virtual Lib compilation. Then, you can check how many bits your ML model would require, which can give you hints as to how it should be modified if you want to compile it to an actual FHE Circuit (not a simulated one) that only supports 8 bits of integer precision.

The Virtual Lib, being pure Python and not requiring crypto key generation, can be much faster than the actual compilation and FHE execution, thus allowing for faster iterations, debugging and FHE simulation, regardless of the bit width used.
