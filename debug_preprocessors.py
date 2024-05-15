import matplotlib.pyplot as plt
import numpy
import numpy as np
from concrete.fhe import Configuration, Integer, univariate

from concrete import fhe
from concrete.ml.common.preprocessors import InsertRounding, TLUDeltaBasedOptimizer, scale_and_round

np.random.seed(42)

# Constant function
def func_const(x):
    x = x.astype(np.float64)
    x = 0.75 * x - 200
    x = x * (x > 0)
    x = x // 118
    # x = (x + 2.1) / 3.4
    x = np.rint(x)
    x = x.astype(np.int64)
    return x


# 2 jumps -> like what we have in CIFAR
def func_2_jump(x):
    x = x.astype(np.float64)
    x = 0.75 * x + 134.0
    x = x * (x > 0)
    x = x // 118
    # x = (x + 2.1) / 3.4
    x = np.rint(x)
    x = x.astype(np.int64)
    return x


# 1 jump
def func_1_jump(x):
    x = x.astype(np.float64)
    x = 0.75 * x + 0.0
    x = x * (x > 0)
    x = x // 118
    # x = (x + 2.1) / 3.4
    x = np.rint(x)
    x = x.astype(np.int64)
    return x


# 5 jumps
def func_5_jump(x):
    x = x.astype(np.float64)
    x = 0.75 * x + 163.0
    x = x * (x > 0)
    x = x // 69
    # x = (x + 2.1) / 3.4
    x = np.rint(x)
    x = x.astype(np.int64)
    return x


def make_step_function(n_thresholds, delta, x_min, x_max, power_of_two=False):
    """Make a step function using a TLU."""
    thresholds_ = []  # First threshold

    if power_of_two:
        th0 = numpy.random.randint(0, delta)
    else:
        th0 = numpy.random.randint(x_min, x_max)

    for index in range(n_thresholds):
        thresholds_.append(th0 + index * delta)

    thresholds = tuple(thresholds_)

    # Step size function to optimize
    def util(x):
        return sum(
            [numpy.where(x >= float(threshold), 1.0, 0.0) for threshold in thresholds]
        )

    def step_function(x):
        return univariate(util)(x).astype(numpy.int64)

    def f(x):
        return step_function(x.astype(numpy.float64))

    def constant_f(x):
        return univariate(lambda x: th0 * (1.0 - (x.astype(numpy.float64) * 0.0)))(
            x
        ).astype(numpy.int64)

    if n_thresholds == 0:
        return constant_f
    return f


def main():
    ok_counter = 0
    nok_counter = 0
    # Activate matplotlib interactive plot
    plt.ion()

    execution_number = 4
    n_bits_from = execution_number + 2

    for bounds in [
        (-(2 ** (n_bits_from)), (2 ** (n_bits_from)) - 1), # 4 ok, 1 nok (func_5)
        (-234, 283), (0, 283), (-283, 284), (-283, 0), (-62, 0), (0, 62),
    ]:  # ]:
        input_range = bounds
        x_min, x_max = bounds
        delta = 2 ** (n_bits_from // 2)  # Constant step size assumption

        for f in [
            make_step_function(execution_number, delta, x_min, x_max, True),
            func_const,
            func_1_jump,
            func_2_jump,
            func_5_jump,
        ]:

            print(f.__name__, bounds)
            inputset = np.arange(input_range[0], input_range[1]+1, 1, dtype=np.int64)
            integer = Integer.that_can_represent(inputset)
            full_range = np.arange(integer.min(), integer.max()+1, 1, dtype=np.int64)
            full_range_to_range_mask = (full_range >= x_min) & (full_range <= x_max)

            def compute(circuit):
                Y = []
                X = []

                for x in full_range:
                    y = circuit.simulate(x)
                    X.append(x)
                    Y.append(y)

                return np.array(Y)

            # Naive
            f_naive = fhe.compiler({"x": "encrypted"})(f)

            circuit_naive = f_naive.compile(inputset)

            naive_res = compute(circuit_naive)


            # Optim - Res
            exactness = fhe.Exactness.EXACT
            optim_exact = TLUDeltaBasedOptimizer(
                overflow_protection=True, exactness=exactness, verbose=1
            )
            cfg_optim_exact = Configuration(additional_pre_processors=[optim_exact])

            f_optim_exact = fhe.compiler({"x": "encrypted"})(f)

            circuit_optim_exact = f_optim_exact.compile(
                inputset, configuration=cfg_optim_exact
            )
            stats = list(optim_exact.statistics.values())[0]

            mult_inputset = scale_and_round(inputset, scaling_factor=stats["scaling_factor"], bias=0, msbs_to_keep=stats["msbs_to_keep"])
            transformed_inputset = scale_and_round(inputset, scaling_factor=stats["scaling_factor"], bias=stats["bias"], msbs_to_keep=stats["msbs_to_keep"])
            y_mult = f(mult_inputset)
            y_transformed = f(transformed_inputset)

            optim_res_exact = compute(circuit_optim_exact)

            optim_exact_error = (
                naive_res[full_range_to_range_mask]
                != optim_res_exact[full_range_to_range_mask]
            ).sum()
            # Plot
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.title(f"Function approximation ({f.__name__}, {bounds=})")
            ax.plot(full_range, naive_res, label="ground-truth", linestyle="--", color="blue", alpha=.5)
            ax.plot(inputset, y_mult, label="scaling_factor", linestyle="-.", color="green", alpha=.5)
            ax.plot(inputset, y_transformed, label="transformed", linestyle="-.", color="orange", alpha=.5)
            ax.plot(
                full_range,
                optim_res_exact,
                label=f"optim-exact err={optim_exact_error}",
                linestyle=":",
                color="red",
                alpha=.5,
            )
            ax.vlines(
                input_range,
                np.min(naive_res),
                np.max(naive_res),
                color="grey",
                linestyle="--",
                label="bounds",
            )
            plt.legend()

            fig, ax = plt.subplots(figsize=(8, 8))
            plt.title(f"Reference - TLU ({f.__name__}, {bounds=})")
            ax.plot(
                full_range,
                naive_res - optim_res_exact,
                label=f"optim-exact err={optim_exact_error}",
                linestyle=":",
            )
            plt.legend()
            plt.draw()

            tol = 1
            if optim_exact_error > tol:
                nok_counter += 1
                plt.pause(0.001)
                breakpoint()
            else:
                ok_counter += 1
                print("OK!")
                plt.pause(0.001)
                plt.close("all")

    print(f"{ok_counter=}, {nok_counter=}")



if __name__ == "__main__":
    main()
