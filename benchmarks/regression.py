import argparse
import json
import random

import py_progress_tracker as progress
from common import (
    REGRESSION_DATASETS,
    REGRESSORS,
    REGRESSORS_STRING_TO_CLASS,
    benchmark_generator,
    benchmark_name_generator,
    seed_everything,
    train_and_test_regressor,
)


def argument_manager():
    # Manage arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir_only", type=int, help="Only dump MLIR (no inference)")
    parser.add_argument("--verbose", action="store_true", help="show more information on stdio")
    parser.add_argument(
        "--datasets",
        choices=REGRESSION_DATASETS,
        type=str,
        nargs="+",
        default=None,
        help="dataset(s) to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 2**32 - 1),
        help="set the seed for reproducibility",
    )
    parser.add_argument(
        "--models",
        choices=REGRESSORS_STRING_TO_CLASS.keys(),
        nargs="+",
        default=None,
        help="regressor(s) to use",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=json.loads,
        default=None,
        help="config(s) to use",
    )
    parser.add_argument(
        "--model_samples",
        type=int,
        default=1,
        help="number of model samples (i.e., overwrite PROGRESS_SAMPLES)",
    )
    parser.add_argument(
        "--fhe_samples", type=int, default=1, help="number of FHE samples on which to predict"
    )
    parser.add_argument(
        "--execute_in_fhe",
        action="store_true",
        default="auto",
        help="force to execute in FHE (default is to use should_test_config_in_fhe function)",
    )
    parser.add_argument(
        "--dont_execute_in_fhe",
        action="store_true",
        help="force to not execute in FHE (default is to use should_test_config_in_fhe function)",
    )
    parser.add_argument(
        "--long_list",
        action="store_true",
        help="just list the different tasks and stop",
    )
    parser.add_argument(
        "--short_list",
        action="store_true",
        help="just list the different tasks (one per model type) and stop",
    )

    args = parser.parse_args()

    if args.dont_execute_in_fhe:
        assert args.execute_in_fhe == "auto"
        args.execute_in_fhe = False

    if args.datasets is None:  # Default to all data-sets
        args.datasets = REGRESSION_DATASETS

    if args.models is None:  # Default to all models
        args.models = REGRESSORS
    else:  # Cast from string to class
        args.models = [REGRESSORS_STRING_TO_CLASS[c] for c in args.models]

    return args


def main():

    # Parameters by the user
    args = argument_manager()

    # Seed everything we can
    seed_everything(args.seed)
    print(f"Using --seed {args.seed}")

    all_tasks = list(benchmark_generator(args))

    # Listing
    if args.long_list or args.short_list:
        already_done_models = {}
        for dataset_i, model_class_i, config_i in all_tasks:
            config_n = json.dumps(config_i).replace("'", '"')
            model_name_i = model_class_i.__name__

            if not args.short_list or model_name_i not in already_done_models:
                print(
                    f"--models {model_name_i} --datasets {dataset_i} "
                    f"--configs '{config_n}' --fhe_samples {args.fhe_samples}"
                )
                already_done_models[model_name_i] = 1
        return

    # Benchmarking / Generating MLIRs
    print(f"Will perform benchmarks on {len(list(all_tasks))} test cases")

    @progress.track(
        [
            {
                "id": benchmark_name_generator(dataset, model, config, "_"),
                "name": benchmark_name_generator(dataset, model, config, " on "),
                "parameters": {"model": model, "dataset": dataset, "config": config},
                "samples": args.model_samples,
            }
            for (dataset, model, config) in all_tasks
        ]
    )
    def perform_regression_benchmark(model, dataset, config):
        """
        This is the test function called by the py-progress module. It just calls the
        benchmark function with the right parameter combination
        """
        train_and_test_regressor(model, dataset, config, args)


if __name__ == "__main__":
    main()
