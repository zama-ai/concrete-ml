"""Script to parse output of pip-audit"""

import argparse
import json
import sys
from pathlib import Path
from typing import List


def format_vulnerability(pkg_name, pkg_version, vuln_info: dict) -> List[str]:
    """Format a vulnerability info.

    Args:
        pkg_name (str): the name of the package
        pkg_version (str): the version of the package
        vuln_info (dict): some information about the vulnerability

    Returns:
        List[str]: A list of formatted vulnerabilities.
    """

    vuln_strs = [
        f"{pkg_name}({pkg_version}) - ID: {vuln['id']} "
        f"fixed in {', '.join(vuln['fix_versions'])}"
        for vuln in vuln_info
    ]
    return vuln_strs


# Cannot have a backslash in f-string, so create a constant for newline
NEW_LINE = "\n"


def main(args):
    """Entry point

    Args:
        args (List[str]): a list of arguments
    """

    vulns_json_path = Path(args.vulns_json).resolve()
    json_content = []
    with open(vulns_json_path, "r", encoding="utf-8") as f:
        json_content.extend(f.readlines())

    report_path = Path(args.vulns_report).resolve()
    has_vulns = False
    potential_vulnerabilities_report = []

    if json_content:
        potential_vulnerabilities_report.append("Found the following vulnerabilities:\n")
        assert len(json_content) == 1
        json_data = json.loads(json_content[0])["dependencies"]
        # print(json.dumps(json_data, indent=4))
        for entry in json_data:
            vuln_entries = entry.get("vulns", [])
            if vuln_entries:
                has_vulns = True
                formatted_vulns = format_vulnerability(
                    entry["name"], entry["version"], vuln_entries
                )
                potential_vulnerabilities_report.append(
                    f"- {f'{NEW_LINE}- '.join(formatted_vulns)}\n"
                )

    with open(report_path, "w", encoding="utf-8") as report:
        if has_vulns:
            report.writelines(potential_vulnerabilities_report)
        else:
            report.write("No vulnerabilities found.\n")

    if has_vulns:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("pip-audit output parser", allow_abbrev=False)

    parser.add_argument(
        "--vulns-json", type=str, required=True, help="The path to the pip-audit json output"
    )
    parser.add_argument(
        "--vulns-report",
        type=str,
        required=True,
        help="Path to the file to which to write the vulneratbility report",
    )

    cli_args = parser.parse_args()
    main(cli_args)
