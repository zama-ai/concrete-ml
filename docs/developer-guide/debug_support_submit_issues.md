# Support and Issues

Concrete ML is a constant work-in-progress, and thus may contain bugs or suboptimal APIs.

Before opening an issue or asking for support, please read this documentation to understand common issues and limitations of Concrete ML. You can also check the [outstanding issues on github](https://github.com/zama-ai/concrete-ml/issues).

Furthermore, undefined behavior may occur if the input-set, which is internally used by the compilation core to set bit-widths of some intermediate data, is not sufficiently representative of the future user inputs. With all the inputs in the input-set, it appears that intermediate data can be represented as an n-bit integer. But, for a particular computation, this same intermediate data needs additional bits to be represented. The FHE execution for this computation will result in an incorrect output, as typically occurs in integer overflows in classical programs.

If you didn't find an answer, you can ask a question on the [Zama forum](https://community.zama.ai) or in the FHE.org [Discord](https://discord.fhe.org).

## Submitting an issue

When submitting an issue ([here](https://github.com/zama-ai/concrete-ml/issues)), ideally include as much information as possible. In addition to the Python script, the following information is useful:

- the reproducibility rate you see on your side
- any insight you might have on the bug
- any workaround you have been able to find

If you would like to contribute to a project and send pull requests, take a look at the [contributor](contributing.md) guide.
