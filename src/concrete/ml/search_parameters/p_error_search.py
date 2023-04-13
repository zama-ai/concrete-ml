"""p_error binary search for classification and regression tasks.

Only PyTorch neural networks and Concrete built-in models are supported.
- Concrete built-in models include trees and QNN
- Quantized aware trained model are supported using Brevitas framework
- Torch models can be converted into post-trained quantized models

The `p_error` represents an essential hyper-parameter in the FHE computation at Zama.
As it impacts the speed of the FHE computations and the model's performance.

In this script, we provide an approach to find out an optimal `p_error`, which would offer
an interesting compromise between speed and efficiency.

The `p_error` represents the probability of a single PBS being incorrect. Know that the FHE
scheme allows to perform 2 types of operations
- Linear operations: additions and multiplications
- Non-linear operation: uni-variate activation functions

At Zama, non-linear operations are represented by table lookup (TLU), which are implemented
through the Programmable Bootstrapping technology (PBS).
A single PBS operation has `p_error` chances of being incorrect.

It's highly recommended to adjust the `p_error` as it is linked to the data-set.

The inference is performed via the FHE simulation mode.

The goal is to look for the largest `p_error_i`, a float ∈ ]0,0.9[, which gives a model_i that has
`accuracy_i`, such that: | accuracy_i - accuracy_0| <= Threshold, where: Threshold ∈ R, given
by the user and `accuracy_0` refers to original model_0 with `p_error_0 ≈ 0.0`.

`p_error` is bounded between 0 and 0.9
`p_error ≈ 0.0`, refers to the original model in clear, that gives an accuracy that we note
as `accuracy_0`.

We assume that the condition is satisfied when we have a match
A match is defined as a uni-variate function, through `strategy` argument, given by the user, it
can be

`any = lambda all_matches: any(all_matches)`
`all = lambda all_matches: all(all_matches)`
`mean = lambda all_matches: numpy.mean(all_matches) >= 0.5`
`median = lambda all_matches: numpy.median(all_matches) == 1`

To validate the results of the FHE simulation and get a stable estimation, we do several simulations
If match, we update the lower bound to be the current `p_error`
Else, we update the upper bound to be the current `p_error`
Update the current `p_error` with the mean of the bounds

We stop the search when the maximum number of iterations is reached.

If we don't reach the convergence, a user warning is raised.
"""
import warnings
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy
import torch
from tqdm import tqdm

from ..common.utils import is_brevitas_model, is_model_class_in_a_list
from ..sklearn import get_sklearn_neural_net_models, get_sklearn_tree_models
from ..torch.compile import compile_brevitas_qat_model, compile_torch_model


def compile_and_simulated_fhe_inference(
    estimator: torch.nn.Module,
    calibration_data: numpy.ndarray,
    ground_truth: numpy.ndarray,
    p_error: float,
    n_bits: int,
    is_qat: bool,
    metric: Callable,
    predict: str,
    **kwargs: Dict,
) -> Tuple[numpy.ndarray, float]:
    """Get the quantized module of a given model in FHE, simulated or not.

    Supported models are:
    - Built-in models, including trees and QNN,
    - Quantized aware trained model are supported using Brevitas framework,
    - Torch models can be converted into post-trained quantized models.

    Args:
        estimator (torch.nn.Module): Torch model or a built-in model
        calibration_data (numpy.ndarray): Calibration data required for compilation
        ground_truth (numpy.ndarray): The ground truth
        p_error (float): Concrete ML uses table lookup (TLU) to represent any non-linear
        n_bits (int): Quantization bits
        is_qat (bool): True, if the NN has been trained through QAT.
            If `False` it is converted into post-trained quantized model.
        metric (Callable): Classification or regression evaluation metric.
        predict (str): The predict method to use.
        kwargs (Dict): Hyper-parameters to use for the metric.

    Returns:
        Tuple[numpy.ndarray, float]: De-quantized or quantized output model depending on
        `is_benchmark_test` and the score.

    Raises:
        ValueError: If the model is neither a built-in model nor a torch neural network.
    """

    compile_params: Dict = {}
    compile_function: Callable[..., Any]
    dequantized_output: numpy.ndarray

    # Custom neural networks with QAT
    if isinstance(estimator, torch.nn.Module):
        if is_qat and is_brevitas_model(estimator):
            compile_function = compile_brevitas_qat_model
        else:
            # Custom neural networks with PTQ
            compile_function = compile_torch_model
            compile_params = {"import_qat": is_qat, "n_bits": n_bits}

        quantized_module = compile_function(
            torch_model=estimator,
            torch_inputset=calibration_data,
            p_error=p_error,
            **compile_params,
        )

        dequantized_output = quantized_module.forward(calibration_data, fhe="simulate")

    elif is_model_class_in_a_list(
        estimator, get_sklearn_neural_net_models() + get_sklearn_tree_models()
    ):
        if not estimator.is_fitted:
            estimator.fit(calibration_data, ground_truth)

        estimator.compile(calibration_data, p_error=p_error)
        predict_method = getattr(estimator, predict)
        dequantized_output = predict_method(calibration_data, fhe="simulate")

    else:
        raise ValueError(
            f"`{type(estimator)}` is not supported. "
            "Supported types are: custom Torch, Brevitas NNs and built-in models (trees and QNNs)."
        )

    score = metric(ground_truth, dequantized_output, **kwargs)

    return dequantized_output, score


# pylint: disable=too-many-instance-attributes, too-many-arguments
class BinarySearch:
    """Class for `p_error` hyper-parameter search for classification and regression tasks."""

    history: List
    path: Path = None  # type: ignore[assignment]
    p_error: float

    def __init__(
        self,
        estimator,
        predict: str,
        metric: Callable,
        n_bits: int = 4,
        is_qat: bool = True,
        lower: float = 0.0,
        upper: float = 0.9,
        max_iter: int = 20,
        n_simulation: int = 5,
        strategy: Any = all,
        max_metric_loss: float = 0.01,
        save: bool = False,
        log_file: str = None,
        directory: str = None,
        verbose: bool = False,
        **kwargs: dict,
    ):
        """`p_error` binary search algorithm.

        Args:
            estimator : Custom model (Brevitas or PyTorch) or built-in models (trees or QNNs).
            predict (str): The prediction method to use for built-in tree models.
            metric (Callable): Evaluation metric for classification or regression tasks.
            n_bits (int): Quantization bits, for PTQ models. Default is 4.
            is_qat (bool): Flag that indicates whether the `estimator` has been trained through
                QAT (quantization-aware training). Default is True.
            lower (float): The lower bound of the search space for the `p_error`. Default is 0.0.
            upper (float): The upper bound of the search space for the `p_error`. Default is 0.9.
                Increasing the upper bound beyond this range may result in longer execution times
                especially when `p_error≈1`.
            max_iter (int): The maximum number of iterations to run the binary search
                algorithm. Default is 20.
            n_simulation (int): The number of simulations to validate the results of the FHE
                simulation. Default is 5.
            strategy (Any): A uni-variate function that defines a "match". It can be built-in
                functions provided in Python, such as any() or all(), or custom functions, like:
                mean = lambda all_matches: numpy.mean(all_matches) >= 0.5
                median = lambda all_matches: numpy.median(all_matches) == 1
                Default is 'all'.
            max_metric_loss (float): The threshold to use to satisfy the condition:
                | accuracy_i - accuracy_0| <= `max_metric_loss`. Default is 0.01.
            save (bool): Flag that indicates whether to save some meta data in log file.
                Default is False.
            log_file (str): The log file name. Default is None.
            directory (str): The directory to save the meta data. Default is None.
            verbose (bool): Flag that indicates whether to print detailed information.
                Default is False.
            kwargs: Parameter of the evaluation metric.
        """

        self.estimator = estimator
        self.n_bits = n_bits
        self.is_qat = is_qat
        self.verbose = verbose
        self.save = save
        self.lower = lower
        self.upper = upper
        self.max_iter = max_iter
        self.n_simulation = n_simulation
        self.max_metric_loss = max_metric_loss
        self.strategy = strategy
        self.metric = metric
        self.predict = predict
        self.kwargs = kwargs

        if directory is not None and log_file is not None:
            self.path = Path(directory) / log_file

        self._check_valid_values()
        self.reset_history()

    def _check_valid_values(self) -> None:
        """Check if the given value is valid according to its attribute."""

        assert 0 <= self.upper <= 1, "Invalid value, `upper` must be between `0` and `1`"
        assert 0 <= self.lower <= 1, "Invalid value, `lower` must be between `0` and `1`"
        assert self.lower < self.upper, "Invalid values, `lower` < `upper`"
        assert (
            self.n_simulation >= 1
        ), "Invalid value, `n_simulation` must be greater or equal than 1"
        assert (
            self.save is True and self.path is not None
        ) or self.save is False, "To save logs, file name and path must be provided"

    def reset_history(self) -> None:
        """Clean history."""
        if self.path is not None:
            self.path.unlink(missing_ok=True)

    def _save(self, data: Dict) -> None:
        """Save data into `directory / log_file.txt`.

        Args:
            data (Dict): Data to be saved.
        """

        if self.save:
            # When instantiating the class, if the log file exists, we reset it
            # On the first iteration, we write the header
            # Then, we append the data at each iteration
            if not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("w", encoding="utf-8") as file:
                    file.write(f"{','.join(data.keys())}\n")  # Iteration = 0, set the header

            # Append new data, as it goes along
            with self.path.open("a", encoding="utf-8") as file:
                file.write(f"{','.join(map(str, data.values()))}\n")

    def _eval(self) -> None:
        """Set the model in an eval mode."""
        if hasattr(self.estimator, "eval"):
            self.estimator.eval()

    @staticmethod
    def eval_match(strategy: Callable, all_matches: List[bool]) -> Union[bool, numpy.bool_]:
        """Eval the matches.

        Args:
            strategy (Callable): A uni-variate function that defines a "match". It can be built-in
                functions provided in Python, such as any() or all(), or custom functions, like:
                mean = lambda all_matches: numpy.mean(all_matches) >= 0.5
                median = lambda all_matches: numpy.median(all_matches) == 1
            all_matches (List[bool]): List of matches.

        Returns:
            bool: Evaluation of the matches according to the given strategy.

        Raises:
            TypeError: If the `strategy` function is not valid.
        """

        try:
            is_match = strategy(all_matches)
            assert isinstance(is_match, (bool, numpy.bool_))
        except (TypeError, AssertionError) as exception:
            raise TypeError(
                "`strategy` is not valid. It takes a list of boolean values as input and "
                "outputs a boolean value. `is_match` must be a boolean."
            ) from exception

        return is_match

    def _acc_diff_objective(
        self,
        reference_output: numpy.ndarray,
        estimated_output: numpy.ndarray,
        reference_score: float,
        estimated_score: float,
    ) -> Dict:
        """Figure out if the selected `p_error` is a good candidate and meets the criteria.

        A good candidate refers to a the largest possible `p_error` that satisfied the following
        condition.

        Considering:
        -  a given threshold ∈ R
        - `p_error = i`, with i ∈ ]0,1[ ∩ R, which gives a model_i that has `accuracy_i`
        - `accuracy_0` refers to original model_0 with `p_error ≈ 0.0`

        | accuracy_i - accuracy_0| <= Threshold and p_error_i > p_error_0

        Args:
            reference_output (numpy.ndarray): The inference of a original model with p_error ≈ 0
            estimated_output (numpy.ndarray): The inference of a model with p_error in ]0,1[
            reference_score (float): The score computed by the original model with p_error ≈ 0
            estimated_score (float): The score computed by the original model with p_error in ]0,1[

        Returns:
            Dict: Set of information
        """

        # The difference between the original model and the model with a given `p_error_i`
        difference = reference_score - estimated_score
        abs_difference = numpy.abs(reference_output - estimated_output)

        # Compute inference errors
        l1_error = abs_difference.mean()
        l2_error = numpy.sqrt(l1_error)
        count_error = (reference_output != estimated_output).sum()
        mean_error = (reference_output != estimated_output).mean()

        # Check if `p_error_i` matches the condition
        match = difference <= self.max_metric_loss

        # Save some metadata
        metadata = OrderedDict(
            {
                "lower": self.lower,
                "p_error": self.p_error,
                "upper": self.upper,
                "l1_error": [round(l1_error, 4)],
                "l2_error": [round(l2_error, 4)],
                "count_error": [count_error],
                "mean_error": [round(mean_error, 4)],
                "estimated_metric": [round(estimated_score, 4)],
                "reference_metric": reference_score,
                "metric_difference": [round(difference, 4)],
                "all_matches": [match],
            }
        )

        self._save(metadata)

        return metadata

    def _update_attr(self, **kwargs: dict) -> None:
        """Update the hyper-parameters then check if the values are valid.

        Args:
            kwargs (Dict): Hyper-parameters

        Raises:
            AttributeError: If the attribute is not in the class.
        """
        for attr, new_value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, new_value)
            else:
                raise AttributeError(f"{attr=} does not belong to this class")

        self._check_valid_values()

    def run(
        self,
        x: numpy.ndarray,
        ground_truth: numpy.ndarray,
        strategy: Callable = all,
        **kwargs: Dict,
    ) -> float:
        """Get an optimal `p_error` using binary search for classification and regression tasks.

        PyTorch models and built-in models are supported.

        To find an optimal `p_error` that  offers a balance between speed and efficiency, we use a
        binary search approach. Where the goal to look for the largest `p_error_i`, a float ∈ ]0,1[,
        which gives a model_i that has `accuracy_i`, such that
        | accuracy_i - accuracy_0| <= max_metric_loss, where max_metric_loss ∈ R and `accuracy_0`
        refers to original model_0 with `p_error ≈ 0.0`.

        We assume that the condition is satisfied when we have a match. A match is defined as a
        uni-variate function, specified through `strategy` argument.

        To validate the results of the FHE simulation and get a stable estimation, we perform
        multiple samplings.
        If match, we update the lower bound to be the current p_error.
        Else, we update the upper bound to be the current p_error.
        Update the current p_error with the mean of the bounds.

        We stop the search either when the maximum number of iterations is reached or when the
        update of the `p_error` is below at a given threshold.

        Args:
            x (numpy.ndarray): Data-set which is used for calibration and evaluation
            ground_truth (numpy.ndarray): The ground truth
            kwargs (Dict): Class parameters
            strategy (Callable): A uni-variate function that defines a "match". It can be: a
                built-in functions provided in Python, like: any or all or a custom function, like:
                mean = lambda all_matches: numpy.mean(all_matches) >= 0.5
                median = lambda all_matches: numpy.median(all_matches) == 1
                Default is `all`.

        Returns:
            float: The optimal `p_error` that aims to speedup computations while maintaining good
                performance.
        """
        self._update_attr(**kwargs)
        self.reset_history()
        self._eval()

        self.history = []

        # Reference predictions:
        # Compile the model in FHE simulation, then compute the score with a model of `p_error ≈ 0`
        reference_output, reference_score = compile_and_simulated_fhe_inference(
            estimator=self.estimator,
            calibration_data=x,
            ground_truth=ground_truth,
            p_error=2**-40,
            is_qat=self.is_qat,
            n_bits=self.n_bits,
            metric=self.metric,
            predict=self.predict,
            **self.kwargs,
        )

        # Set `p_error`
        self.p_error = (self.lower + self.upper) / 2.0

        # Binary search algorithm
        for _ in tqdm(range(self.max_iter), disable=not self.verbose):
            # Run the inference with a given p-error

            # Since `p_error` represents a probability, to validate the results of the Fhe
            # simulation and get a stable estimation, several runs are needed
            simulation_data = []
            for _ in range(self.n_simulation):
                current_output, current_score = compile_and_simulated_fhe_inference(
                    estimator=self.estimator,
                    calibration_data=x,
                    ground_truth=ground_truth,
                    p_error=self.p_error,
                    is_qat=self.is_qat,
                    n_bits=self.n_bits,
                    metric=self.metric,
                    predict=self.predict,
                    **self.kwargs,
                )

                current_metadata = self._acc_diff_objective(
                    reference_output=reference_output,
                    estimated_output=current_output,
                    reference_score=reference_score,
                    estimated_score=current_score,
                )

                simulation_data.append(current_metadata)

            # Aggregating all metadata collected from `n` simulations to display them all at once
            self.history.append(
                OrderedDict(
                    {
                        k: sum((d[k] for d in simulation_data), [])
                        if isinstance(simulation_data[0][k], list)
                        else simulation_data[0][k]
                        for k in simulation_data[0]
                    }
                )
            )

            if self.verbose:
                pprint(self.history[-1])

            # Update interval
            if self.eval_match(strategy, self.history[-1]["all_matches"]):
                # If we valid our criteria, we increase the `p_error`
                self.lower = self.p_error
            else:
                # If not, we decrease the `p_error`
                self.upper = self.p_error
            self.p_error = (self.lower + self.upper) / 2.0

        # Raise a user warning if the convergence is not reached
        if numpy.mean(self.history[-1]["metric_difference"]) > self.max_metric_loss:
            # pylint: disable=pointless-statement
            warning_message = (
                "ConvergenceWarning: The convergence is not reached for the "
                f"`{self.estimator.__class__}` model, you can:\n"
                "  - Increase the number of iterations `max_iter` or `n_simulation`.\n"
                "  - Choose another strategy.\n"
                "  - Increase the size of the calibration data-set."
            )
            warnings.simplefilter("always")
            warnings.warn(warning_message, category=UserWarning, stacklevel=2)

        return self.p_error
