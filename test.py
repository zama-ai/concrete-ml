from concrete.ml.common.preprocessors import TLUDeltaBasedOptimizer, InsertRounding
from concrete import fhe
import numpy as np
import matplotlib.pyplot as plt
from concrete.fhe import Configuration, Integer

input_range = (-234, 283)

inputset = np.arange(input_range[0], input_range[1], dtype=np.int64)
integer = Integer.that_can_represent(inputset)
full_range = np.arange(integer.min(), integer.max(), dtype=np.int64)

# Constant function
def f(x):
    x = x.astype(np.float64)
    x = 0.75 * x - 200
    x = x * (x > 0)
    x = x // 118
    # x = (x + 2.1) / 3.4
    x = np.rint(x)
    x = x.astype(np.int64)
    return x

# 2 jumps -> like what we have in CIFAR
def f(x):
    x = x.astype(np.float64)
    x = 0.75 * x + 134.
    x = x * (x > 0)
    x = x // 118
    # x = (x + 2.1) / 3.4
    x = np.rint(x)
    x = x.astype(np.int64)
    return x

# 1 jump
def f(x):
    x = x.astype(np.float64)
    x = 0.75 * x + 0.
    x = x * (x > 0)
    x = x // 118
    # x = (x + 2.1) / 3.4
    x = np.rint(x)
    x = x.astype(np.int64)
    return x

# 5 jumps
def f(x):
    x = x.astype(np.float64)
    x = 0.75 * x + 163.
    x = x * (x > 0)
    x = x // 69
    # x = (x + 2.1) / 3.4
    x = np.rint(x)
    x = x.astype(np.int64)
    return x

# TODO: check with f that has non-constant delta

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

# Optim - Approx
exactness = fhe.Exactness.APPROXIMATE
optim = TLUDeltaBasedOptimizer(overflow_protection=True, exactness=exactness, internal_bit_width_target=18)
pre_proc_optim = [optim]
cfg_optim = Configuration(additional_pre_processors=pre_proc_optim)

f_optim = fhe.compiler({"x": "encrypted"})(f)

circuit_optim = f_optim.compile(inputset, configuration=cfg_optim)

optim_res = compute(circuit_optim)

# Optim - Res
exactness = fhe.Exactness.EXACT
optim_exact = TLUDeltaBasedOptimizer(overflow_protection=True, exactness=exactness, internal_bit_width_target=18)
pre_proc_optim_exact = [optim_exact]
cfg_optim_exact = Configuration(additional_pre_processors=pre_proc_optim_exact)

f_optim_exact = fhe.compiler({"x": "encrypted"})(f)

circuit_optim_exact = f_optim_exact.compile(inputset, configuration=cfg_optim_exact)

optim_res_exact = compute(circuit_optim_exact)

# Round (bit-width-from-optim)
n_bits_round = list(optim.statistics.values())[0]["optimized_bitwidth"] if optim.statistics else None
rounding_from_optim_no_scaling = InsertRounding(n_bits_round, overflow_protection=True)
pre_proc_round_from_optim_no_scaling = [rounding_from_optim_no_scaling]
cfg_round_from_optim_no_scaling = Configuration(additional_pre_processors=pre_proc_round_from_optim_no_scaling)

f_round_from_optim_no_scaling = fhe.compiler({"x": "encrypted"})(f)

circuit_round_from_optim_no_scaling = f_round_from_optim_no_scaling.compile(inputset, configuration=cfg_round_from_optim_no_scaling)

round_res_from_optim_no_scaling = compute(circuit_round_from_optim_no_scaling)

# Plot
fig, ax = plt.subplots()
ax.plot(full_range, naive_res,label="ground-truth", linestyle="-")
ax.plot(full_range, optim_res,label="optim-approx", linestyle="-.")
ax.plot(full_range, optim_res_exact,label="optim-exact", linestyle=":")
ax.plot(full_range, round_res_from_optim_no_scaling,label="round-from-optim", linestyle="-.")
ax.vlines(input_range, np.min(naive_res), np.max(naive_res), color="grey", linestyle="--", label="bounds")
plt.legend()

# Set the secondary ticks
if optim.statistics:
    lsbs_to_remove = integer.bit_width - list(optim.statistics.values())[0]["optimized_bitwidth"]
    rounded_ticks = full_range[np.concatenate(
        [
             
            np.diff(
                fhe.round_bit_pattern(full_range, lsbs_to_remove=lsbs_to_remove)
            ).astype(bool),
            np.array([False,]), 
        ]
    )]
    
    # Create secondary axes for the top ticks
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())  # Make sure the secondary axis has the same limits as the primary axis
    
    # Set the secondary ticks
    ax_top.set_xticks(rounded_ticks)
    
    # Customize appearance of secondary ticks
    ax_top.tick_params(which='minor', length=4, color='red')

plt.show()
