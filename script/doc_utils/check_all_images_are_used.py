"""Script that checks if images are used in some files."""

import argparse
import multiprocessing
import sys
from functools import partial
from pathlib import Path


def check_image_is_used(image, files):
    """Checks that the image is used somewhere in the given files.

    Args:
        image (str): The image to consider
        files (List[str]): The files to check

    Returns:
        bool: If the image is used at least once in the given files
    """
    image_path = Path(image)

    # Iterate over all the given files
    for file_path in files:
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as file:

            # Iterate over all the file's lines
            for line in file:
                line = line.rstrip()

                # If the image's name is found in the line, return True
                if image_path.name in line:
                    return True

    # The Print the image's path as its currently not used in the project
    print("Image is not used:", image_path)

    return False


def main(args):
    """Entry point.

    Args:
        args (List[str]): a list of arguments
    """

    print("Starting to check for unused images in the given files\n")

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res = pool.map(partial(check_image_is_used, files=args.files), args.images)

        # Count the total number of unused images within the given files
        unused_images = len(res) - sum(res)

        if unused_images == 0:
            print("All images are used at least once")
        else:
            print(f"\nA total of {unused_images} images are never used")

        # Exit 0 if all images are used, else 1
        final_status = unused_images != 0

        sys.exit(final_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", type=str, nargs="+", required=True, help="The images to check")
    parser.add_argument(
        "--files", type=str, nargs="+", required=True, help="The files to modify in place"
    )

    main(parser.parse_args())
