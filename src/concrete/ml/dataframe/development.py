import itertools
import json
from pathlib import Path

from concrete import fhe


def identity_pbs(a):
    """Define an identity TLU."""
    return fhe.univariate(lambda x: x)(a)


@fhe.compiler(
    {"val_1": "encrypted", "val_2": "encrypted", "left_key": "encrypted", "right_key": "encrypted"}
)
def encrypted_sum_on_condition(val_1, val_2, left_key, right_key):
    """Atomic function to compose during the left join.

    Args:
        val_1 (int): Value 1 to sum
        val_2 (int): Value 2 to sum
        left_key (int): Left key to match
        right_key (int): Right key to match

    Returns:
        int: Summed value
    """
    condition = left_key == right_key

    sum_on_condition = val_1 + (val_2 * condition)

    sum_with_tlu = identity_pbs(sum_on_condition)

    return sum_with_tlu


def get_merge_inputset(n_bits: int):
    """Define the inputset to use for the merge operator."""
    # Build the circuit using at most `n_bits` bits, which defines :
    # - the input's unsigned integer dtype allowed (at most)
    # - the max number of rows possible in an input (assuming the merge is done on a column of
    # unsigned integers starting at 1)
    # Note that a 0 is used to represent a NaN (Not a Number) value
    max_row = high = 2**n_bits - 1

    # TODO: we should not use the data-frame's number of rows as this should not be known
    inputset = list(
        itertools.product(
            [0, high],
            [0, high],
            [1, max_row],
            [1, max_row],
        )
    )

    return inputset


# Link the circuits and inputset generators to their operator's name
OPS_TO_COMPILATION_CONFIG = {
    "merge": {
        "inputset_func": get_merge_inputset,
        "cp_func": encrypted_sum_on_condition,
    }
}


def get_circuit(op_name, n_bits: int):
    """Get the compiled FHE circuit from the given operator."""
    inputset = OPS_TO_COMPILATION_CONFIG[op_name]["inputset_func"](n_bits)
    cp_func = OPS_TO_COMPILATION_CONFIG[op_name]["cp_func"]

    # Compile for composability
    merge_circuit = cp_func.compile(inputset, composable=True)

    return merge_circuit


def save_deployment(deployment_dir: Path, op_name: str, n_bits: int):
    """Save the client and server files for the given operator."""

    # Define the dtype using the given n_bits
    # TODO: This assumes that we only support unsigned integers with a certain bit-width
    dtype = f"uint{n_bits}"

    op_dtype_dir = deployment_dir / f"{op_name}/{dtype}"

    # If the directory does not exist/is empty
    if not op_dtype_dir.is_dir() or not any(op_dtype_dir.iterdir()):
        op_dtype_dir.mkdir(parents=True, exist_ok=True)

        merge_circuit = get_circuit(op_name, n_bits)

        # First save the circuit for the server
        server_path = op_dtype_dir / "server.zip"
        merge_circuit.server.save(server_path, via_mlir=True)

        # Save the circuit for the client
        client_path = op_dtype_dir / "client.zip"
        merge_circuit.client.save(client_path)

        metadata = {
            "n_bits": n_bits,
            "dtype": dtype,
        }

        # Save metadata file
        metadata_path = op_dtype_dir / "metadata.json"
        with metadata_path.open("w") as metadata_file:
            json.dump(metadata, metadata_file)
