import datetime
import requests
import argparse
import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def generate_figure(data, path_to_csv: Path, path_to_png: Path):
    number_of_runs = data["total_count"]
    runs = data["workflow_runs"]

    # Should be equal
    print(number_of_runs, len(runs))

    # Cast to DataFrame
    df = pd.DataFrame(runs)
    for col in ["updated_at", "created_at", "run_started_at"]:
        values = df[col]
        assert isinstance(values, pd.Series)
        df[col] = pd.to_datetime(values, unit="ns")

    # Copy
    sub_df = df.copy()

    # Filtering
    sub_df = sub_df[sub_df["conclusion"] == "success"]
    assert isinstance(sub_df, pd.DataFrame)
    sub_df = sub_df[sub_df["status"] == "completed"]
    assert isinstance(sub_df, pd.DataFrame)
    sub_df = sub_df[sub_df["name"].apply(lambda elt: elt in {'CML builds (weekly or not)', 'CML weekly builds', 'concrete-ml CI Pipeline'})]
    assert isinstance(sub_df, pd.DataFrame)

    # 4 week history
    now = datetime.datetime.now()
    start_date = pd.Timestamp(now, tz="UTC") - pd.Timedelta(weeks=4)

    # Filter based on starting date
    sub_df = sub_df[sub_df["run_started_at"] >= start_date]
    assert isinstance(sub_df, pd.DataFrame)
    # Handle dates
    sub_df["time_diff"] = (sub_df["updated_at"] - sub_df["run_started_at"]).astype(int)
    assert isinstance(sub_df, pd.DataFrame)
    # ns to seconds
    sub_df["time_diff"] = sub_df["time_diff"] / 1e9
    assert isinstance(sub_df, pd.DataFrame)
    # seconds to minutes
    sub_df["time_diff"] = sub_df["time_diff"] / 60
    assert isinstance(sub_df, pd.DataFrame)
    # Compute relative time (for regression)
    sub_df["relative_time"] = (sub_df["run_started_at"] - sub_df["run_started_at"].min())
    assert isinstance(sub_df, pd.DataFrame)

    sub_df.to_csv(str(path_to_csv.resolve()))

    # Just to generate the regression line
    regressor = LinearRegression()
    X = sub_df[["relative_time"]].view(int)
    regressor.fit(X=X, y=sub_df["time_diff"])
    y_pred = regressor.predict(X)

    # Generate figure
    fig, ax = plt.subplots(figsize=(8, 4), dpi=800)
    fig.suptitle("Successful CML-CI time execution in minutes over the 4 last weeks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Minutes")
    # ax.set_yscale("log")
    ax.scatter(sub_df["run_started_at"], sub_df["time_diff"], marker="x")
    ax.plot(sub_df["run_started_at"], y_pred, color="red")
    plt.xticks(rotation='vertical')
    plt.tight_layout()

    # Save figure
    fig.savefig(str(path_to_png.resolve()))


def get_data(token: str, path_to_json: Path):
    """Get all information from workflow runs from github for CML-internal"""
    # Create session with correct headers
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    })

    per_page = 100  # 100 is the maximum per the documentation
    index = 1
    # Get only jobs that targeted main
    params = {
        "branch": "main", 
        "per_page": per_page, 
        "page": index,
        "event": "push",
    }
    user = "zama-ai"
    repo = "concrete-ml-internal"
    final_result = dict()

    while True:
        # Get number of runs
        response = session.get(
            url=f"https://api.github.com/repos/{user}/{repo}/actions/runs",
            params=params
        )
        json_response = response.json()

        if index == 1:
            final_result = json_response
            print("Total:", int(json_response["total_count"]))
        else:
            final_result["workflow_runs"] += json_response["workflow_runs"]

        runs_count = len(json_response["workflow_runs"])
        print(runs_count)
        print({
            "current_length": len(final_result["workflow_runs"])
        })
        if runs_count == 0:
            break
        index += 1
        params["page"] = index

    # Dump
    with path_to_json.open(mode="w", encoding="utf-8") as file:
        json.dump(obj=final_result, fp=file, indent=4)

    return final_result


def main(token: str, path_to_json, path_to_csv, path_to_png):
    data = get_data(token=token, path_to_json=path_to_json)
    generate_figure(data, path_to_csv=path_to_csv, path_to_png=path_to_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    gh_token_default = {}
    if "GH_TOKEN" in os.environ:
        gh_token_default["default"] = os.environ["GH_TOKEN"]
    parser.add_argument("--gh-token", dest="token", type=str, **gh_token_default)
    parser.add_argument(
        "--path-to-dump-json",
        dest="path_to_dump_json",
        default=Path(os.getcwd()) / "github_ci_data.json",
        type=lambda elt: Path(elt).resolve()
    )
    parser.add_argument(
        "--path-to-dump-csv",
        dest="path_to_dump_csv",
        default=Path(os.getcwd()) / "github_ci_data.csv",
        type=lambda elt: Path(elt).resolve()
    )
    parser.add_argument(
        "--path-to-dump-png",
        dest="path_to_dump_png",
        default=Path(os.getcwd()) / "cml_ci_time_evolution.png",
        type=lambda elt: Path(elt).resolve()
    )
    args = parser.parse_args()
    main(token=args.token, path_to_json=args.path_to_dump_json, path_to_csv=args.path_to_dump_csv, path_to_png=args.path_to_dump_png)
