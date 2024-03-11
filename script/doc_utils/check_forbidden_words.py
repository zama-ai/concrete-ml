"""Check forbidden words in our files, to have more coherent file or avoid English mistakes"""

import argparse
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union


def check_forbidden_construction(
    forbidden_word_construction: str,
    forbidden_word: str,
    line: str,
    file: Union[str, Path],
    nline: int,
) -> bool:
    """Check forbidden construction in a line

    Args:
        forbidden_word_construction (str): forbidden word construction
        forbidden_word (str): forbidden word
        line (str): line
        file (Union[str, Path]): file
        nline (int): line number

    Returns:
        bool: False if forbidden_word_construction in line
    """
    if forbidden_word_construction in line:
        print(f"-> `{forbidden_word}` in {file}:{nline}={line}")
        return False

    return True


def process_file(file_str: str, do_open_problematic_files=False):
    """Check forbidden words in a file

    Args:
        file_str (str): path to file as string
        do_open_problematic_files (bool): if files with problem will be open or not

    Returns:
        Tuple[bool, int]: if everything is okay, and the number of errors

    """
    file_path = Path(file_str).resolve()

    # Don't do it on API or advanced example data files
    if "docs/references/api" in file_str or "docs/advanced_examples/data" in file_str:
        return True, 0

    # Don't do it for conventions.md file that explains what should or not be done
    if file_path.name == "conventions.md":
        return True, 0

    # forbidden_word_list is a list of tuples: each tuple is of the form
    # (forbidden_word, exceptions, excepted_file_types), where:
    #       forbidden_word: is the forbidden word
    #       exceptions: is a list (possibly empty) of exceptions; if the forbidden_word is found
    #           but one of the elements in the exceptions match the string, it is not considered as
    #           an error
    #       excepted_file_types: is a list of possibly empty file types for which this forbidden
    #           word is ignored
    forbidden_word_list: List[Tuple[str, List, List[str]]]
    forbidden_word_list = [
        ("Concrete-ml", [], []),  # use `Concrete ML`
        ("Concrete-Ml", [], []),  # use `Concrete ML`
        ("Concrete-ML", [], []),  # use `Concrete ML`
        ("concrete ml", [], []),  # use `Concrete ML`
        ("concrete-ml", [], []),  # use `Concrete ML`
        ("pytorch", [], []),  # use `PyTorch`
        ("Pytorch", [], []),  # use `PyTorch`
        ("pytorch", [], []),  # use `PyTorch`
        ("bitwidth", [], [".py"]),  # use `bit-width`
        ("bit width", [], []),  # use `bit-width`
        ("inputset", [], [".py"]),  # use `input-set`
        ("dataset", [], [".py"]),  # use `data-set`
        ("datasets", [], [".py"]),  # use `data-sets`
        ("data set", [], []),  # use `data-set`
        ("data sets", [], []),  # use `data-sets`
        ("data-base", [], []),  # use `database`
        ("code-base", [], []),  # use `codebase`
        ("dequantize", [], []),  # use de-quantize
        ("dequantization", [], []),  # use de-quantization
        ("requantize", [], []),  # use re-quantize
        ("a FHE", [], []),  # use `an FHE`
        ("can Google", [], []),  # google is a verb
        ("jupyter", [], []),  # use Jupyter
        ("PyTest", [], []),  # use pytest
        ("pyTest", [], []),  # use pytest
        ("Pytest", [], []),  # use pytest
        ("python", ["python client.py", "python ./server.py", "python -m"], []),  # use Python
        ("HummingBird", [], []),  # use Hummingbird
        (
            "hummingbird",
            ["from hummingbird import", "import hummingbird", "from hummingbird."],
            [],
        ),  # use Hummingbird
        ("MacOS", [], []),  # use macOS
        ("macos", [], []),  # use macOS
        ("MacOs", [], []),  # use macOS
        ("bareOS", [], []),  # use bare OS
        ("BareOS", [], []),  # use bare OS
        ("torch", ["import torch", "torch.", "from torch import"], [".py"]),  # use Torch
        ("numpy", ["import numpy", "from numpy import", "numpy."], [".py"]),  # use NumPy
        ("Numpy", [], [".py"]),  # use NumPy
        ("PoissonRegression", [], []),  # use Poisson Regression
        ("docker", ["docker ps -a"], []),  # Use Docker
        ("poetry", [], []),  # Use Poetry
        ("Make", [], [".py"]),  # Use make
        (
            "brevitas",
            ["import brevitas", "from brevitas", "bit accuracy brevitas"],
            [".py"],
        ),  # use Brevitas
        ("concrete-numpy", [], []),  # use Concrete
        ("concrete-Numpy", [], []),  # use Concrete
        ("Concrete-numpy", [], []),  # use Concrete
        ("Concrete-Numpy", [], []),  # use Concrete
        ("concrete numpy", [], []),  # use Concrete
        ("concrete Numpy", [], []),  # use Concrete
        ("Concrete numpy", [], []),  # use Concrete
        ("Concrete Numpy", [], []),  # use Concrete
        ("cnp", [], []),  # use fhe (or cp, worst case)
        ("tool-kit", [], []),  # use toolkit
        ("tool-kits", [], []),  # use toolkits
        (
            "preprocessing",
            ["import preprocessing", "preprocessing."],
            [".py"],
        ),  # use pre-processing
        ("preprocess", [], []),  # use pre-process
        ("keras", [], []),  # use Keras
        ("tensorflow", ["= tensorflow."], []),  # use TensorFlow
        ("Tensorflow", [], []),  # use TensorFlow
        ("gauss", [], []),  # use Gauss
        ("gaussian", [], []),  # use Gaussian
        ("netron", [], []),  # use Netron
        ("information are", [], []),  # information is
        ("builtin", [], []),  # use built-in
        ("hyper parameters", [], []),  # use hyper-parameters
        ("hyperparameters", [], []),  # use hyper-parameters
        ("combinaison", [], []),  # use combination
        ("zeropoint", [], []),  # use zero-point
        ("pretrained", [], []),  # use pre-trained
        ("i.e.", ["i.e.,"], []),  # use i.e.,
        ("e.g.", ["e.g.,"], []),  # use e.g.,
        ("discord", [], []),  # use Discord
        ("worst-case", [], []),  # use worst case
        ("FHE friendly", [], []),  # use FHE-friendly
        ("slow-down", [], []),  # use slow down
        ("counter-part", [], []),  # use counterpart
        ("Scikit-learn", [], ["README.md"]),  # use scikit-learn
        ("Scikit-Learn", [], []),  # use scikit-learn
        ("scikit-Learn", [], []),  # use scikit-learn
        ("it's", [], []),  # use `it is`
        ("It's", [], []),  # use `It is`
        ("let's", [], []),  # keep a consistent impersonal style
        ("Let's", [], []),  # keep a consistent impersonal style
        (
            "let us",
            ["feel free to let us know"],
            ["README.md"],
        ),  # keep a consistent impersonal style
        ("Let us", [], []),  # keep a consistent impersonal style
        ("github", [], []),  # use GitHub
        ("elementwise", [], []),  # use element-wise
        ("favourite", [], []),  # use favorite
        (
            "speed up",
            ["to speed up", "will speed up", "will not speed up", "it speeds up", "this speeds up"],
            [".py"],
        ),
        ("de-activate", [], []),  # use deactivate
        ("Skorch", [], []),  # use skorch
        ("fhe", ["execute_in_fhe", "forward_fhe", "fhe_circuit", "fhe.org"], [".py"]),  # use `FHE`
        ("tradeoff", [], []),  # use trade-off
        ("th", [], []),  # use the
        ("appropriat", [], []),  # use appropriate
        ("constrains", [], []),  # use constraints
        ("CML", [], []),  # use Concrete ML
        ("CN", ["CNN"], []),  # use Concrete Python
        ("CP", [], []),  # use Concrete Python
        ("ie", [], []),  # use i.e.,
        ("ie,", [], []),  # use i.e.,
        ("ie.,", [], []),  # use i.e.,
        ("eg", [], []),  # use e.g.,
        ("eg,", [], []),  # use e.g.,
        ("eg., ", [], []),  # use e.g.,
    ]
    # For later
    #   "We" or "Our", or more generally, passive form

    is_everything_ok = True
    nb_errors = 0

    with open(file_path, "r", encoding="utf-8") as f:

        nline = 0

        for line in f:

            line = line.rstrip()

            for forbidden_word, exceptions, excepted_file_types in forbidden_word_list:

                stop = False

                # Exceptions on file types
                for t in excepted_file_types:
                    if t in str(file_path):
                        stop = True
                        break

                # Exceptions for the given line
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
    """Entry point.

    Args:
        args (Namespace): list of arguments
    """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res = pool.map(partial(process_file, do_open_problematic_files=args.open), args.files)
        res_first = [r[0] for r in res]
        res_second = [r[1] for r in res]

        # Exit 0 if all went well as True == 1
        final_status = not all(res_first)

        if final_status != 0:
            print(f"Number of errors: {sum(res_second)}")

        sys.exit(final_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--files", type=str, nargs="+", required=True, help="The files to modify in place"
    )

    parser.add_argument("--open", action="store_true", help="Open files with problems for edit")

    cli_args = parser.parse_args()
    main(cli_args)
