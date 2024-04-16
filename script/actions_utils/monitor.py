"""Module to generate figure of evolution of Concrete ML-CI time on main for last 4 weeks."""

import argparse
import datetime
import json
import os
import typing
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression


@typing.no_type_check
def generate_figure(data: Dict[str, Any], path_to_csv: Path, path_to_png: Path):
    """Generate the figure of timing evolution on last 4 weeks using GitHub API data.

    Arguments:
        data (Dict[str, Any]): a dict with the results of the Github API
        path_to_csv (Path): path to csv where to dump filtered data
        path_to_png (Path): path to png of the figure
    """
    number_of_runs = data["total_count"]
    runs = data["workflow_runs"]

    # Should be equal
    print(number_of_runs, len(runs))

    # Cast to DataFrame
    dataframe = pd.DataFrame(runs)
    for col in ["updated_at", "created_at", "run_started_at"]:
        values = dataframe[col]
        assert isinstance(values, pd.Series)
        dataframe[col] = pd.to_datetime(values, unit="ns")

    # Copy
    sub_df = dataframe.copy()

    # Filtering
    sub_df = sub_df[sub_df["conclusion"] == "success"]
    assert isinstance(sub_df, pd.DataFrame)
    sub_df = sub_df[sub_df["status"] == "completed"]
    assert isinstance(sub_df, pd.DataFrame)
    mask = sub_df["name"].apply(
        lambda elt: elt
        in {"CML builds (weekly or not)", "CML weekly builds", "concrete-ml CI Pipeline"}
    )
    assert isinstance(mask, pd.Series)
    sub_df = sub_df[mask]
    assert isinstance(sub_df, pd.DataFrame)

    # 4 week history
    now = datetime.datetime.now()
    start_date = pd.Timestamp(now, tz="UTC") - pd.Timedelta(weeks=4)

    # Filter based on starting date
    mask = sub_df["run_started_at"] >= start_date
    assert isinstance(mask, pd.Series)
    sub_df = sub_df[mask]
    # Handle dates
    sub_df["time_diff"] = (sub_df["updated_at"] - sub_df["run_started_at"]).astype(int)
    # Nanoseconds to seconds
    sub_df["time_diff"] = sub_df["time_diff"] / 1e9
    # seconds to minutes
    sub_df["time_diff"] = sub_df["time_diff"] / 60
    # Compute relative time (for regression)
    sub_df["relative_time"] = sub_df["run_started_at"] - sub_df["run_started_at"].min()
    assert isinstance(sub_df, pd.DataFrame)

    sub_df.to_csv(str(path_to_csv.resolve()))

    # Just to generate the regression line
    regressor = LinearRegression()
    x_train_df = sub_df[["relative_time"]]
    assert isinstance(x_train_df, pd.DataFrame)
    x_train = x_train_df.astype(int)
    regressor.fit(X=x_train, y=sub_df["time_diff"])
    y_pred = regressor.predict(x_train)

    # Generate figure
    fig, axis = plt.subplots(figsize=(8, 4), dpi=800)
    assert isinstance(axis, Axes)

    fig.suptitle("Successful Concrete ML-CI time execution in minutes over the 4 last weeks")
    axis.set_xlabel("Date")
    axis.set_ylabel("Minutes")
    # ax.set_yscale("log")
    axis.scatter(sub_df["run_started_at"], sub_df["time_diff"], marker="x")
    axis.plot(sub_df["run_started_at"], y_pred, color="red")
    plt.xticks(rotation="vertical")
    plt.tight_layout()

    # Save figure
    fig.savefig(str(path_to_png.resolve()))


def get_data(token: str, path_to_json: Path):
    """Get all information from workflow runs from github for the concrete-ml repo.

    Arguments:
        token (str): Github API token
        path_to_json: path to json file where to dump raw result of query

    Returns:
        Dict[str, Any]: a dict with the results of the Github API
    """
    # Create session with correct headers
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
        }
    )

    per_page = 100  # 100 is the maximum per the documentation
    index = 1
    # Get only jobs that targeted main
    params: Dict[str, Union[str, int]] = {
        "branch": "main",
        "per_page": per_page,
        "page": index,
        "event": "push",
    }
    user = "zama-ai"
    repo = "concrete-ml"
    final_result = {}

    while True:

        # Get number of runs
        response = session.get(
            url=f"https://api.github.com/repos/{user}/{repo}/actions/runs", params=params
        )

        json_response = response.json()

        if index == 1:
            final_result = json_response
            print("Total:", int(json_response["total_count"]))
        else:
            final_result["workflow_runs"] += json_response["workflow_runs"]

        runs_count = len(json_response["workflow_runs"])
        print(runs_count)
        print({"current_length": len(final_result["workflow_runs"])})
        if runs_count == 0:
            break
        index += 1
        params["page"] = index

    # Dump
    with path_to_json.open(mode="w", encoding="utf-8") as file:
        json.dump(obj=final_result, fp=file, indent=4)

    return final_result


def main(token: str, path_to_json, path_to_csv, path_to_png):
    """Main function to get data and generate figure.

    Arguments:
        token (str): Github API token
        path_to_json: path to json file where to dump raw result of query
        path_to_csv (Path): path to csv where to dump filtered data
        path_to_png (Path): path to png of the figure
    """
    data = get_data(token=token, path_to_json=path_to_json)
    generate_figure(data, path_to_csv=path_to_csv, path_to_png=path_to_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    if "GH_TOKEN" in os.environ:
        parser.add_argument("--gh-token", dest="token", type=str, default=os.environ["GH_TOKEN"])
    else:
        parser.add_argument("--gh-token", dest="token", type=str)
    parser.add_argument(
        "--path-to-dump-json",
        dest="path_to_dump_json",
        default=Path(os.getcwd()) / "github_ci_data.json",
        type=lambda elt: Path(elt).resolve(),
    )
    parser.add_argument(
        "--path-to-dump-csv",
        dest="path_to_dump_csv",
        default=Path(os.getcwd()) / "github_ci_data.csv",
        type=lambda elt: Path(elt).resolve(),
    )
    parser.add_argument(
        "--path-to-dump-png",
        dest="path_to_dump_png",
        default=Path(os.getcwd()) / "cml_ci_time_evolution.png",
        type=lambda elt: Path(elt).resolve(),
    )
    args = parser.parse_args()
    main(
        token=args.token,
        path_to_json=args.path_to_dump_json,
        path_to_csv=args.path_to_dump_csv,
        path_to_png=args.path_to_dump_png,
    )
