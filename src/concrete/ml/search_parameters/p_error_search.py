# Make it more generic
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3246

"""p_error binary search for classification tasks.

Only Pytorch neural networks are supported so far.
- Quantized aware trained model are supported using Brevitas framework
- Torch models can be converted into post-trained quantized models

The `p_error` represents an essential hyper-parameter in the FHE computation at Zama.
As it impacts the speed of the FHE computations and the model's performance.

In this script, we provide an approach to find out an optimal `p_error`, which would offer
an interesting compromise between speed and efficiency.

The `p_error` represents the probability of a single PBS being incorrect. Know that the FHE
scheme allows to perfom 2 types of operations
- Linear operations: additions and multiplications
- Non-linear operation: uni-variate activation functions

At Zama, non-linear operations are represented by table lookup (TLU), which are implemented
through the Programmable Bootstrapping technology (PBS).
A single PBS operation has `p_error` chances of being incorrect.

It's highly recommended to adjust the `p_error` as it is linked to the data-set.

The inference is performed via the FHE simulation mode also known as Virtual Library (VL)

The goal is to look for the largest `p_error_i`, a float ∈ ]0,1[, which gives a model_i that has
`accuracy_i`, such that: | accuracy_i - accuracy_0| <= Threshold, where: Threshold ∈ R, given
by the user and `accuracy_0` refers to original model_0 with `p_error_0 = 0.0`.

`p_error` is bounded between 0 and 1
`p_error = 0.0`, refers to the original model in clear, that gives an accuracy that we note
as `accuracy_0`
`p_error = 1.0`, refers to the worst case scenario, where the model perfoms very badly
By default, `lower = 0.0` and `uppder = 1.0`.

We assume that the condition is satisfied when we have a match
A match is defined as a univariate function, through `strategy` argument, given by the user, it
can be

`any = lambda all_matches: any(all_matches)`
`all = lambda all_matches: all(all_matches)`
`mean = lambda all_matches: numpy.mean(all_matches) >= 0.5`
`median = lambda all_matches: numpy.median(all_matches) == 1`

To validate the results of the VL and get a stable estimation, we do several simulations
If match, we update the lower bound to be the current p_error
Else, we update the upper bound to be the current p_error
Update the current p_error with the mean of the bounds

We stop the search either when the maximum number of iterations is reached or when the update of
the `p_error` is below at a given threshold

If we don't reach the convergence, a user warning is raised.
"""
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy
import torch
from concrete.numpy import Configuration
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

from ..common.utils import is_brevitas_model
from ..quantization import QuantizedModule
from ..torch.compile import compile_brevitas_qat_model, compile_torch_model


def get_quantized_module(
    *,
    estimator: torch.nn.Module,
    calibration_data: numpy.ndarray,
    p_error: float,
    is_qat: bool = True,
) -> QuantizedModule:
    """Get the quantized module of a given model in FHE, simulated or not.

    Supported models are:
    - Quantized aware trained model are supported using Brevitas framework,
    - Torch models can be converted into post-trained quantized models.

    Args:
        estimator (torch.nn.Module): Torch model
        calibration_data (numpy.ndarray): Calibration data required for compilation
        p_error (float): Concrete-ML uses table lookup (TLU) to represent any non-linear
        is_qat (bool): True, if the NN has been trained through QAT.
            If `False` it is converted into post-trained quantized model.

    Returns:
        QuantizedModule: Quantized numpy module.

    Raises:
        ValueError: If the NN is not QAT.
    """

    configuration = Configuration(
        # This is for our tests in Virtual Library only
        dump_artifacts_on_unexpected_failures=True,
        enable_unsafe_features=True,
    )

    compile_params: Dict = {}
    compile_function: Callable[..., Any]

    if is_qat is True and is_brevitas_model(estimator):
        compile_function = compile_brevitas_qat_model

    elif isinstance(estimator, torch.nn.Module):
        compile_function = compile_torch_model
        compile_params = {"import_qat": is_qat}
    else:
        raise ValueError(
            f"`{type(estimator)}` is not supported. Please use a Torch or a Brevitas neural network"
        )

    quantized_module = compile_function(
        torch_model=estimator,
        torch_inputset=calibration_data,
        configuration=configuration,
        use_virtual_lib=True,
        p_error=p_error,
        **compile_params,
    )

    return quantized_module


def simulated_fhe_inference(*, X: numpy.ndarray, quantized_module) -> numpy.ndarray:
    """Run the inference in FHE simulated mode.

    Args:
        X (numpy.ndarray): Data
        quantized_module: Quantized numpy module.

    Returns:
        numpy.ndarray: Inference.
    """
    quantized_x = quantized_module.quantize_input(X)
    quantized_output = quantized_module.forward_in_fhe(quantized_x, simulate=True)
    dequantized_output = quantized_module.dequantize_output(quantized_output)

    return dequantized_output


# pylint: disable=too-many-instance-attributes
class BinarySearch:
    """Class for `p_error` hyper-parameter search for classification tasks."""

    history: List = []
    p_error: float
    quantized_module: QuantizedModule

    def __init__(
        self,
        *,
        estimator,
        is_qat: bool = True,
        verbose: bool = True,
        save: bool = False,
        lower: float = 0.0,
        upper: float = 1.0,
        max_iter: int = 100,
        n_simulation: int = 10,
        strategy: Any = all,
        max_metric_loss: float = 0.01,
        delta_tolerence: float = 1e-5,
        log_file: str = "log_file.txt",
        directory: str = "/tmp/cml_search/p_error",
    ):
        """`p_error` binary search algorithm.

        Args:
            estimator : QAT torch NN
            is_qat (bool): Flag that indicates whether the `estimator` has been
                trained through QAT (quantization-aware training). Default is True
            verbose (bool): Flag that indicates whether to print detailed information.
                Default is True.
            save (bool): Flag that indicates whether to save some meta data in
                `log_file.txt`file. Default is False.
            lower (float): The lower bound of the search space for the `p_error`.
                Default is 0.0.
            upper (float): The upper bound of the search space for the `p_error`.
                Default is 1.0.
            max_iter (int): The maximum number of iterations to run the binary search
                algorithm. Default is 100.
            n_simulation (int): The number of simulations to validate the results of
                the VL. Default is 10.
            max_metric_loss (float): The threshold to use to satisfy the condition:
                | accuracy_i - accuracy_0| <= `max_metric_loss` and stop the search algorithm.
                Default is 0.01.
            strategy (Any): A uni-variate function that defines a "match". It can be:
                a built-in functions provided in python, like:
                any or all or a custom function, like:
                mean = lambda all_matches: numpy.mean(all_matches) >= 0.5
                median = lambda all_matches: numpy.median(all_matches) == 1
                Default is 'all'.
            log_file (str): The kog file name. Default is 'log_file.txt'.
            delta_tolerence (float): Tolerance's threshold of the relative difference between
                |current_p_error - previous_p_error|. Default is 1e-5.
            directory (str): The directory to save the meta data.
                Default is '/tmp/cml_search/p_error'.

        """

        self.estimator = estimator
        self.is_qat = is_qat
        self.verbose = verbose
        self.save = save
        self.lower = lower
        self.upper = upper
        self.max_iter = max_iter
        self.n_simulation = n_simulation
        self.max_metric_loss = max_metric_loss
        self.strategy = strategy
        self.log_file = log_file
        self.delta_tolerence = delta_tolerence
        self.directory = Path(directory)

        self._check_valid_values()
        self.reset_history()

    def _check_valid_values(self) -> None:
        """Check if the given value is valid according to its attribute."""

        assert 0 <= self.upper <= 1, "Invalid value, `upper` must be between `0` and `1`"
        assert 0 <= self.lower <= 1, "Invalid value, `lower` must be between `0` and `1`"
        assert self.lower < self.upper, "Invalid values, `lower` < `upper`"
        assert (
            self.n_simulation >= 1
        ), "Invalid value, `n_evaluation` must be greater or equal than 1"

    def reset_history(self) -> None:
        """Clean history."""
        full_path = self.directory / self.log_file
        if full_path.exists():
            full_path.unlink(missing_ok=True)

    def _save(self, data: Dict) -> None:
        """Save data into `self.dir / self.log_file.txt`.

        Args:
            data (Dict): Data to be saved.
        """

        self.history.append(data)

        if self.save:
            # When instantiating the class, if `self.log_file` exists, we reset it
            # On the first iteration, we write the header
            # Then, we append the data at each iteration
            if not (self.directory / self.log_file).exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                with open(self.directory / self.log_file, "w", encoding="utf-8") as file:
                    file.write(f"{','.join(data.keys())}\n")  # Iteration = 0, set the header
                    file.write(f"{','.join(map(str, data.values()))}\n")
            else:
                # Append new data, as it goes along
                with open(self.directory / self.log_file, "a", encoding="utf-8") as file:
                    file.write(f"{','.join(map(str, data.values()))}\n")

    def _eval(self) -> None:
        """Set the model in an eval mode."""
        if hasattr(self.estimator, "eval"):
            self.estimator.eval()

    @staticmethod
    def eval_match(strategy: Callable, all_match: List) -> bool:
        """Eval the matches.

        Args:
            strategy (Callable): A uni-variate function that defines a "match". It can be: a
                built-in functions provided in python, like: any or all or a custom function, like:
                mean = lambda all_matches: numpy.mean(all_matches) >= 0.5
                median = lambda all_matches: numpy.median(all_matches) == 1
            all_match (List): List of matches.

        Returns:
            bool: Evaluation of the matches according to the given strategy.

        Raises:
            TypeError: If the `strategy` function is not valid.
        """

        if not isinstance(strategy, Callable):  # type: ignore[arg-type]
            raise TypeError(f"`{strategy}` is not valid.")

        return strategy(all_match)

    def _acc_diff_objective(
        self,
        reference: numpy.ndarray,
        reference_accuracy: float,
        model_output: numpy.ndarray,
        ground_truth: numpy.ndarray,
        k: int = 1,
    ) -> bool:
        """Figure out if the selected `p_error` is a good candidate and meets the criteria.

        A good candidate refers to a the largest possible `p_error` that satisfied the following
        condition.

        Considering:
        -  a given threshold ∈ R
        - `p_error = i`, with i ∈ ]0,1[ ∩ R, which gives a model_i that has `accuracy_i`
        - `accuracy_0` refers to original model_0 with `p_error = 0.0`

        | accuracy_i - accuracy_0| <= Threshold and p_error_i > p_error_0

        Args:
            model_output (QuantizedModule): Quantized module
            reference (numpy.ndarray): The inference computed by the original model
            reference_accuracy (float): The accuracy computed by the original model
            ground_truth (numpy.ndarray): The targets
            k (int): Number of most likely outcomes considered to find the correct label.

        Returns:
            bool: if it matches and some information
        """

        # Compute the accuracy for a selected `p_error = i`
        estimated_accuracy = top_k_accuracy_score(
            y_true=ground_truth,
            y_score=model_output,
            k=k,
            labels=numpy.arange(model_output.shape[-1]),
        )

        # The difference between the original model and the model with a given `p_error`
        difference = abs(reference_accuracy - estimated_accuracy)
        abs_difference = numpy.abs(reference - model_output)

        # Compute inference errors
        l_1_error = abs_difference.sum()
        l_inf_error = abs_difference.max()
        count_error = (reference != model_output).sum()
        mean_error = (reference != model_output).mean()

        # Check if `p_error_i` matches the condition
        match = difference <= self.max_metric_loss

        # Save some meta data
        meta_output = {
            "p_error": self.p_error,
            "low": self.lower,
            "upper": self.upper,
            "accuracy_difference": difference,
            "match": match,
            "l1_error": l_1_error,
            "linf_error": l_inf_error,
            "count_error": count_error,
            "mean_error": mean_error,
            "strategy": self.strategy,
            "estimated_accuracy": estimated_accuracy,
            "reference_accuracy": reference_accuracy,
        }

        self._save(meta_output)

        return match

    def _update_attr(self, **kwargs: dict) -> None:
        """Update the hyper-parameters then check if the values are valid.

        Args:
            kwargs (Dict): Hyper-parameters

        Raises:
            AttributeError: If the attribut is not in the class.
        """
        for attr, new_value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, new_value)
            else:
                raise AttributeError(f"{attr=} does not belong to this class")

        self._check_valid_values()

    def run(
        self, x: numpy.ndarray, ground_truth: numpy.ndarray, strategy: Callable, **kwargs: Dict
    ) -> float:
        """Get an optimal `p_error` using binary search method for classification tasks.

        Only PyTorch models are supported. If the given model is not quantization-aware trained, it
        will be converted into a post-trained quantized model.

        To find an optimal `p_error` that  offers a balance between speed and efficiency, we use a
        binary search approach. Where the goal to look for the largest `p_error_i`, a float ∈ ]0,1[,
        which gives a model_i that has `accuracy_i`, such that
        | accuracy_i - accuracy_0| <= max_metric_loss, where max_metric_loss ∈ R and `accuracy_0`
        refers to original model_0 with `p_error = 0.0`.

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
                built-in functions provided in python, like: any or all or a custom function, like:
                mean = lambda all_matches: numpy.mean(all_matches) >= 0.5
                median = lambda all_matches: numpy.median(all_matches) == 1

        Returns:
            float: The optimal `p_error` that aims to speedup computations while maintaining good
                performance.
        """
        self._eval()

        self._update_attr(**kwargs)
        self.reset_history()

        # Reference predictions:
        # `p_error = 0.0`, corresponds to the original model in clear
        quantized_module = get_quantized_module(
            estimator=self.estimator,
            calibration_data=x,
            p_error=0.0,
            is_qat=self.is_qat,
        )

        # Compute the reference accuracy obtained with the original model with `p_error = 0`
        reference_output = simulated_fhe_inference(quantized_module=quantized_module, X=x)
        reference_accuracy = top_k_accuracy_score(
            y_true=ground_truth,
            y_score=reference_output,
            labels=numpy.arange(reference_output.shape[-1]),
            k=1,
        )
        # Set `p_error`
        self.p_error = (self.lower + self.upper) / 2.0

        # Binary search algorithm
        for _ in tqdm(range(self.max_iter), disable=self.verbose is None):
            # Run the inference with given p-error
            # Run predictions
            current_quantized_module = get_quantized_module(
                estimator=self.estimator,
                calibration_data=x,
                p_error=self.p_error,
                is_qat=self.is_qat,
            )

            # Since `p_error` represents a probability, to validate the results of the VL and get
            # a stable estimation, several simulations are needed
            all_matches = []
            for _ in tqdm(range(self.n_simulation), disable=self.verbose is None):
                current_output = simulated_fhe_inference(
                    quantized_module=current_quantized_module, X=x
                )

                is_matched = self._acc_diff_objective(
                    model_output=current_output,
                    reference=reference_output,
                    reference_accuracy=reference_accuracy,
                    ground_truth=ground_truth,
                )

                all_matches.append(is_matched)

            # Update update interval
            if self.eval_match(strategy, all_matches):
                # If we valid our criteria, we increase the `p_error`
                self.lower = self.p_error
            else:
                # If not, we decrease the `p_error`
                self.upper = self.p_error
            previous_p_error = self.p_error
            self.p_error = (self.lower + self.upper) / 2

            if self.verbose:
                print(f"{self.history[-1]}\n")

            # If |previous_perror - current_perror | <= threshold, we consider that the convergence
            # is reached and we stop the search
            if (
                abs(self.p_error - previous_p_error) <= self.delta_tolerence
                or self.history[-1]["accuracy_difference"] <= self.max_metric_loss
            ):
                if self.verbose:
                    print("Convergence reached")
                break

        # Raise a user warning if the convergence is not reached
        if self.history[-1]["accuracy_difference"] > self.max_metric_loss:
            # pylint: disable=pointless-statement
            warning_message = "The convergence is not reached. You can:"
            f"You can increase the number `{self.max_iter=}` or `{self.n_simulation=}\n"
            f"Choose another `{self.strategy=}`\nIncrease the size of the calibration dataset."
            warnings.warn(warning_message, category=UserWarning, stacklevel=2)

        return self.p_error
