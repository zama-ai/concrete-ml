import itertools
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy

from concrete import fhe
from concrete.ml.quantization.quantized_module import QuantizedArray


# The simplest tokenizer there is
def tokenize(x):
    return x.split()


def one_hot(x: List[str], vocabulary, seq_length):
    assert len(x) <= seq_length
    vocab_size = len(vocabulary)
    representation = numpy.zeros((seq_length, vocab_size), dtype=numpy.int64)
    representation[:, 0] = 1  # unk-padding
    for index, word in zip(range(seq_length), x):
        representation[index, vocabulary[word]] = 1
        representation[index, 0] = 0  # remove padding
    return representation


def token_index(x: List[str], vocabulary, seq_length):
    assert len(x) <= seq_length
    representation = numpy.zeros((seq_length,), dtype=numpy.int64)  # 0 matches unk-pad token
    for index, word in zip(range(seq_length), x):
        representation[index] = vocabulary[word]
    return representation


def main():
    # Llama: https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md#quantitative-analysis
    # embedding dimension between 4096 and 8192 with 512 by default in the code
    # sequence length between 512, 1024 and 2048
    # vocabulary size always of 32_000
    vocabulary_sizes = [32_000]
    seq_lengths = [512]
    embedding_dimensions = [4096]
    n_bits = [8]

    metrics_path = Path("metrics.csv")
    metrics_path.unlink(missing_ok=True)
    metrics_path.touch()
    print("Metrics in:", metrics_path.resolve())
    with metrics_path.open("a", encoding="utf-8") as file:
        metrics = [
            "compile-time",
            "keygen-time",
            "fhe-run-time",
            "encryption-time",
            "serialization-time",
            "ciphertext-size",
        ]
        metrics_names = list(
            itertools.chain(
                *[[f"{method}_{metric}" for metric in metrics] for method in ["mat-mult", "tlu"]]
            )
        ) + ["sequence-length", "vocabulary-size", "embedding-dimension", "n-bit"]
        file.write(",".join(metrics_names) + "\n")

    for seq_length, vocabulary_size, embedding_dimension, n_bit in itertools.product(
        vocabulary_sizes, seq_lengths, embedding_dimensions, n_bits
    ):
        print()
        metric_values: Dict[str, float] = {
            "sequence-length": seq_length,
            "vocabulary-size": vocabulary_size,
            "embedding-dimension": embedding_dimension,
            "n-bit": n_bit,
        }
        print(metric_values)
        method = "mat-mult"
        print(f"{method} ...")
        # Define objects
        vocabulary = {str(i): i for i in range(vocabulary_size)}
        embedding_matrix = numpy.array(
            numpy.random.randint(
                low=0, high=embedding_dimension, size=(vocabulary_size, embedding_dimension)
            ),
            dtype=numpy.float64,
        )
        vocabulary_ = list(vocabulary.keys())
        corpus = [" ".join(random.choice(vocabulary_) for _ in range(seq_length))]
        print(embedding_matrix.shape)

        print("quantizing ...")
        start = time.time()
        quantized_embedding = QuantizedArray(values=embedding_matrix, n_bits=n_bit).qvalues
        end = time.time()
        duration = end - start
        print(f"done in {duration} seconds")

        # One-Hot + Mat-mult embedding approach
        print("one-hot encoding ...")
        start = time.time()
        inputs_one_hot = (
            one_hot(x=tokenize(x=corpus[0]), vocabulary=vocabulary, seq_length=seq_length),
        )
        end = time.time()
        duration = end - start
        print(f"done in {duration} seconds")

        def embedding(x):
            # pylint: disable-next=cell-var-from-loop
            return x @ quantized_embedding  # noqa: B023

        # Clear
        print("Computing in clear ...")
        start = time.time()
        function_output_mat_mult = embedding(*inputs_one_hot)
        end = time.time()
        duration = end - start
        print(f"done in {duration} seconds")

        # FHE
        print("compiling ...")
        fhe_embedding = fhe.compiler({"x": "encrypted"})(embedding)
        start = time.time()
        circuit_mat_mult = fhe_embedding.compile(
            [
                one_hot(x=tokenize(x=sentence), vocabulary=vocabulary, seq_length=seq_length)
                for sentence in corpus
            ]
        )
        end = time.time()
        compile_time = end - start
        print(f"done in {compile_time} seconds")
        metric_values[f"{method}_compile-time"] = compile_time

        print("keygen ...")
        start = time.time()
        circuit_mat_mult.keygen(force=True)
        end = time.time()
        keygen_time = end - start
        print(f"done in {keygen_time} seconds")
        metric_values[f"{method}_keygen-time"] = keygen_time

        print("running in FHE", end=" ")
        start = time.time()
        circuit_output_mat_mult = circuit_mat_mult.encrypt_run_decrypt(*inputs_one_hot)
        end = time.time()
        fherun_time = end - start
        print(f"done in {fherun_time} seconds")
        metric_values[f"{method}_fhe-run-time"] = fherun_time

        print("encrypting ...")
        start = time.time()
        ciphertext = circuit_mat_mult.client.encrypt(inputs_one_hot[0])
        end = time.time()
        encryption_time = end - start
        print(f"done in {encryption_time} seconds")
        metric_values[f"{method}_encryption-time"] = encryption_time

        print("serializing ...")
        start = time.time()
        serialized_args_matmult: bytes = ciphertext.serialize()
        end = time.time()
        serialization_time = end - start
        print(f"done in {serialization_time} seconds")
        metric_values[f"{method}_serialization-time"] = serialization_time

        ciphertext_size = len(serialized_args_matmult) / 1_000_000_000
        metric_values[f"{method}_ciphertext-size"] = ciphertext_size

        print("Cipher-Text size (Giga-bytes):", ciphertext_size)
        print("Mat-Mult FHE==Clear:", (circuit_output_mat_mult == function_output_mat_mult).all())
        del circuit_mat_mult
        del circuit_output_mat_mult
        del ciphertext
        del ciphertext_size
        del serialized_args_matmult

        # TLU approach (broadcast + multi-TLU)
        # Reasonable ciphertext but slow
        # (lots of PBs, same PBS applied a lot -> more than 8 bit would benefit from acceleration)
        method = "tlu"
        print(f"{method} ...")
        inputs_token = (
            token_index(x=tokenize(x=corpus[0]), vocabulary=vocabulary, seq_length=seq_length),
        )

        embeddings_tlu = fhe.LookupTable(
            [
                [  # embed_{embed_dim}[token_index]
                    fhe.LookupTable(quantized_embedding[:, embed_dim])
                    for embed_dim in range(quantized_embedding.shape[1])
                ]
                for _ in range(seq_length)
            ]
        )  # seq_length x embedding_dimension x vocabulary_size

        embedding_dimension = quantized_embedding.shape[1]

        def tlu_embedding(x):
            # pylint: disable-next=cell-var-from-loop
            return embeddings_tlu[  # noqa: B023
                numpy.broadcast_to(
                    numpy.expand_dims(x, axis=1),  # seq_length
                    # pylint: disable-next=cell-var-from-loop
                    (seq_length, embedding_dimension),  # noqa: B023
                )
            ]  # noqa: B023

        # Clear
        function_output_tlu = tlu_embedding(*inputs_token)

        # FHE
        print("compiling ...")
        fhe_embedding = fhe.compiler({"x": "encrypted"})(tlu_embedding)
        start = time.time()
        circuit_tlu = fhe_embedding.compile(
            [
                token_index(x=tokenize(x=sentence), vocabulary=vocabulary, seq_length=seq_length)
                for sentence in corpus
            ]
        )
        end = time.time()
        compile_time = end - start
        print(f"done in {compile_time} seconds")
        metric_values[f"{method}_compile-time"] = compile_time

        print("keygen ...")
        start = time.time()
        circuit_tlu.keygen(force=True)
        end = time.time()
        keygen_time = end - start
        print(f"done in {keygen_time} seconds")
        metric_values[f"{method}_keygen-time"] = keygen_time

        print("fhe run ...")
        start = time.time()
        circuit_output_tlu = circuit_tlu.encrypt_run_decrypt(*inputs_token)
        end = time.time()
        fherun_time = end - start
        print(f"done in {fherun_time} seconds")
        metric_values[f"{method}_fhe-run-time"] = fherun_time

        print("encrypting ...")
        start = time.time()
        ciphertext = circuit_tlu.client.encrypt(inputs_token[0])
        end = time.time()
        encryption_time = end - start
        print(f"done in {encryption_time} seconds")
        metric_values[f"{method}_encryption-time"] = encryption_time

        print("serializing ...")
        start = time.time()
        serialized_args_tlu: bytes = ciphertext.serialize()
        end = time.time()
        serialization_time = end - start
        print(f"done in {serialization_time} seconds")
        metric_values[f"{method}_serialization-time"] = serialization_time

        ciphertext_size = len(serialized_args_tlu) / 1_000_000_000
        metric_values[f"{method}_ciphertext-size"] = ciphertext_size

        print("Cipher-Text size (Giga-bytes):", ciphertext_size)
        print("TLU FHE==Clear:", (circuit_output_tlu == function_output_tlu).all())

        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(",".join(str(metric_values[key]) for key in metrics_names) + "\n")


if __name__ == "__main__":
    main()
