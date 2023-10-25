# General imports
import argparse
import warnings
from time import time

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from concrete.ml.sklearn import _get_sklearn_tree_models, is_classifier_or_partial_classifier

# pylint: disable=E0611
from concrete.ml.sklearn.utils_test_tree_rounding import (
    check_if_file_exists,
    set_lstr,
    set_seed,
    write,
)

warnings.filterwarnings("ignore")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Update the ROUNDING variable in X.py.")
parser.add_argument("n_bits", type=int, help="x")
parser.add_argument("lsbr", type=int, help="x")
parser.add_argument("--seed", type=int, help="random seed", default=42)
parser.add_argument("--file_name", type=str, default="impact_rounding_on_tree_based_models.txt")

args = parser.parse_args()

lsbr = args.lsbr
seed = args.seed
n_bits = args.n_bits
file_name = args.file_name
models = _get_sklearn_tree_models()

set_seed(seed)
set_lstr(lsbr)
check_if_file_exists(file_name)

print(f"{n_bits=} | {lsbr=} | {seed=}")

for model_class in tqdm(models):
    for ds in [1500, 5000]:
        for features in [5, 100, 200]:
            model = model_class(n_bits=n_bits)
            X, y = make_classification(n_samples=ds, n_features=features, random_state=seed)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=seed
            )
            model, sklearn_model = model.fit_benchmark(X_train, y_train)

            write(file_name, f"{model.__class__.__name__},{ds},{features},")

            circuit = model.compile(X_train)
            max_bit = circuit.graph.maximum_integer_bit_width()

            write(file_name, f"{n_bits},{lsbr},{max_bit},")

            start_time = time()
            circuit.keygen()
            delta_time_key = time() - start_time

            write(file_name, f"{delta_time_key:.5f},")

            y_pred_sk = sklearn_model.predict(X_test)
            y_pred_simulate = model.predict(X_test, fhe="simulate")
            y_pred_disable = model.predict(X_test, fhe="disable")

            metric = accuracy_score if is_classifier_or_partial_classifier(model) else r2_score

            score_sk = metric(y_true=y_test, y_pred=y_pred_sk)
            score_disable = metric(y_true=y_test, y_pred=y_pred_disable)
            score_simulate = metric(y_true=y_test, y_pred=y_pred_simulate)

            write(file_name, f"{score_disable:.5f},{score_simulate:.5f},")

            start_time = time()
            y_pred_execute = model.predict(X_test[0, None], fhe="execute")
            delta_time_inf = time() - start_time
            score_execute = (y_pred_execute == y_test[0]).mean()

            write(file_name, f"{delta_time_inf:.5f},{score_sk:.5f},{seed}\n")

            print(
                f"{model.__class__.__name__}, {ds=}, dim={features}, {n_bits=}, {lsbr=}, {max_bit=},"
                f"sk={score_sk:.3f}, disable={score_disable:.3f}, simulate={score_simulate:.3f}, "
                f"key_gen={delta_time_key:.2f}, {delta_time_inf=:.2f}\n"
            )
