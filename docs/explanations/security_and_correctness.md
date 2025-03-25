# Security and correctness

## Security model

The default parameters for Concrete ML are chosen considering the [IND-CPA](https://en.wikipedia.org/wiki/Ciphertext_indistinguishability) security model, and are selected with a [bootstrapping off-by-one error probability](../explanations/advanced_features.md#tolerance-to-off-by-one-error-for-an-individual-tlu) of $$2^-40$$. In particular, it is assumed that the results of decrypted computations are not shared by the secret key owner with any third parties, as such an action can lead to leakage of the secret encryption key. If you are designing an application where decryptions must be shared, you will need to craft custom encryption parameters which are chosen in consideration of the IND-CPA^D security model \[1\].

## Correctness of computations

The [cryptography concepts](../getting-started/concepts.md#cryptography-concepts) section explains how Concrete ML can ensure **guaranteed correctness of encrypted computations**. In this approach, a quantized machine learning model will be converted to an FHE circuit that produces the same result on encrypted data as the original model on clear data.

However, the [bootstrapping off-by-one error probability](../explanations/advanced_features.md#tolerance-to-off-by-one-error-for-an-individual-tlu) can be configured by the user. Raising this probability results in lower latency when executing on encrypted data, but higher values cancel the correctness guarantee of the default setting. In practice this may not be an issue, as the accuracy of the model may be maintained, even though slight differences are observed in the model outputs. Moreover, as noted in the [paragraph above](#security-model), raising the off-by-one error probability may negatively impact the security model.

Furthermore, a second approach to reduce latency at the expense of correctness is approximate computation of univariate functions. This mode is enabled by using the [rounding setting](../explanations/advanced_features.md#rounded-activations-and-quantizers). When using the [`fhe.Exactness.APPROXIMATE`](../references/api/concrete.ml.torch.compile.md#function-compile_torch_model) rounding method, off-by-one errors are always induced in the computation of activation functions, irrespective of the bootstrapping off-by-one error probability.

When trading-off better latency for correctness, it is highly recommended to use the [FHE simulation feature](../getting-started/concepts.md#i-model-development) to measure accuracy on a drawn-out test-set. In many cases the accuracy of the model is only slightly impacted by approximate computations.

## References

\[1\] Li, Baiyu, et al. “Securing approximate homomorphic encryption using differential privacy.” Annual International Cryptology Conference. Cham: Springer Nature Switzerland, 2022. https://eprint.iacr.org/2022/816.pdf
