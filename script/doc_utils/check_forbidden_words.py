"""."""

import argparse
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path
from typing import List, Tuple


def check_forbidden_construction(forbidden_word_construction, forbidden_word, line, file, nline):
    """Check forbidden construction in a line"""
    if forbidden_word_construction in line:
        print(f"-> `{forbidden_word}` in {file}:{nline}={line}")
        return False

    return True


def process_file(file_str: str, do_open_problematic_files=False):
    """Check forbidden words in a file"""
    file_path = Path(file_str).resolve()

    # Don't do it on API files
    if "docs/developer-guide/api" in file_str:
        return True, 0

    # forbidden_word_list is a list of tuples: each tuple is of the form
    # (forbidden_word, exceptions), where:
    #       forbidden_word: is the forbidden word
    #       exceptions: is a list (possibly empty) of exceptions; if the forbidden_word is found
    #           but one of the elements in the exceptions match the string, it is not considered as
    #           an error
    forbidden_word_list: List[Tuple[str, List]]
    forbidden_word_list = [
        ("Concrete-ml", []),
        ("Concrete-Ml", []),
        ("Concrete ML", []),
        ("pytorch", []),
        ("bitwidth", []),
        ("bit width", []),
        ("inputset", []),
        ("dataset", []),
        ("data-base", []),
        ("code-base", []),
        ("dequantize", []),
        ("requantize", []),
        ("an FHE", []),
        ("can Google", []),
        ("can not", []),
        ("jupyter", []),
        ("PyTest", []),
        ("pyTest", []),
        ("Pytest", []),
        ("python", []),
        ("HummingBird", []),
        ("hummingbird", ["from hummingbird import", "import hummingbird", "from hummingbird."]),
        ("MacOS", []),
        ("macos", []),
        ("MacOs", []),
        ("bareOS", []),
        ("BareOS", []),
        ("torch", ["import torch", "torch.", "from torch import"]),
        ("numpy", ["import numpy", "from numpy import", "numpy."]),
        ("Numpy", ["Concrete-Numpy"]),
        ("PoissonRegression", []),
        ("docker", []),
        ("poetry", []),
        ("Make", []),
        ("brevitas", ["import brevitas", "from brevitas", "bit accuracy brevitas"]),
        ("concrete-numpy", []),
        ("tool-kit", []),
        ("tool-kits", []),
        ("preprocessing", []),
        ("preprocess", []),
        ("keras", []),
        ("tensorflow", ["= tensorflow."]),
        ("Tensorflow", []),
        ("gauss", []),
        ("gaussian", []),
        ("netron", []),
        ("information are", []),
        ("builtin", []),
        ("hyper parameters", []),
        ("hyperparameters", []),
    ]
    # For later
    #   "We" or "Our", or more generally, passive form

    is_everything_ok = True
    nb_errors = 0

    with open(file_path, "r", encoding="utf-8") as f:

        nline = 0

        for line in f:

            line = line.rstrip()

            for forbidden_word, exceptions in forbidden_word_list:

                stop = False

                for exception in exceptions:
                    if exception in line:
                        stop = True
                        break

                if stop:
                    continue

                for forbidden_word_construction in [
                    f" {forbidden_word} ",
                    f" {forbidden_word}.",
                    f" {forbidden_word},",
                    f" {forbidden_word}:",
                    f"({forbidden_word} ",
                    f"[{forbidden_word} ",
                    f"[{forbidden_word}]",
                ]:

                    local_check = check_forbidden_construction(
                        forbidden_word_construction, forbidden_word, line, file_path, nline
                    )

                    nb_errors += 1 - int(local_check)
                    is_everything_ok &= local_check

            nline += 1

    if not is_everything_ok and do_open_problematic_files:
        os.system(f"subl {file_path}")

    return is_everything_ok, nb_errors


def main(args):
    """Entry point."""
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res = pool.map(partial(process_file), args.files)
        res_first = [r[0] for r in res]
        res_second = [r[1] for r in res]

        # Exit 0 if all went well as True == 1
        final_status = not all(res_first)

        if final_status != 0:
            print(f"Number of errors: {sum(res_second)}")

        sys.exit(final_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FIXME", allow_abbrev=False)

    parser.add_argument(
        "--files", type=str, nargs="+", required=True, help="The files to modify in place"
    )

    cli_args = parser.parse_args()
    main(cli_args)
