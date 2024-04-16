"""Finalize Jupyter notebooks."""

import argparse
import json
from pathlib import Path


# pylint: disable=too-many-nested-blocks
def main():
    """Finalize

    Raises:
        Exception: if notebooks are not sanitized.
    """

    parser = argparse.ArgumentParser(description="Sanitizer for Jupyter Notebooks")

    parser.add_argument("base", type=str, nargs="+", help="directory which contains the notebooks")
    parser.add_argument("--check", action="store_true", help="flag to enable just checking mode")

    args = parser.parse_args()

    for directory in args.base:
        base = Path(directory)
        notebooks = base.glob("**/*.ipynb")

        for notebook in notebooks:
            path = str(notebook)

            if "_build" in path or ".ipynb_checkpoints" in path:
                continue

            with open(notebook, "r", encoding="utf-8") as f:
                content = json.load(f)

            if args.check:
                try:
                    metadata = content["metadata"]
                    assert len(metadata) == 1
                    assert "execution" in metadata

                    execution = metadata["execution"]
                    assert len(execution) == 1
                    assert "timeout" in execution

                    timeout = execution["timeout"]
                    assert timeout == 10800  # 3 hours

                    for cell in content["cells"]:
                        if "metadata" in cell:
                            assert len(cell["metadata"]) == 0

                except Exception as exception:
                    message = "Notebooks are not sanitized. Please run `make conformance`."
                    print(message)
                    raise Exception(message) from exception
            else:
                content["metadata"] = {
                    "execution": {
                        "timeout": 10800,  # 3 hours
                    }
                }

                for cell in content["cells"]:
                    if "metadata" in cell:
                        cell["metadata"] = {}

                with open(notebook, "w", newline="\n", encoding="utf-8") as f:
                    json.dump(content, f, indent=1, ensure_ascii=False)
                    f.write("\n")


if __name__ == "__main__":
    main()
