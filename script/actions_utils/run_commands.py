"""Script to run commands from a json file"""

import argparse
import json
import subprocess


def main():
    """Main function: runs a list of commands from a json file"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", dest="json_file", type=str, default="./commands.json")
    arguments = parser.parse_args()
    with open(arguments.json_file, mode="r", encoding="utf-8") as file:
        commands = json.load(file)
    assert isinstance(commands, list)
    assert all(isinstance(elt, str) for elt in commands)
    for command in commands:
        command = command.replace("{", "'{").replace("}", "}'")
        print()
        print("Command:")
        print(command)
        print()
        try:
            success_output = subprocess.check_output(command, shell=True).decode("utf-8")
        except subprocess.CalledProcessError as exception:
            print("Error:")
            print(exception)
        else:
            print("Success:")
            print(success_output)
        print()


if __name__ == "__main__":
    main()
