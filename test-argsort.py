import numpy as np
import jax
from jax import numpy as jnp
from jax import random

key = random.PRNGKey(42)

def _run_argsort_numpy(args) -> jnp.ndarray:
    jax.debug.print("Run argsort using Numpy")
    arr, axis = args
    return np.argsort(arr, axis=axis).astype(np.int32)

def run_argsort_numpy(arr:jnp.ndarray, axis=None) -> jnp.ndarray:
    if jax.devices()[0].device_kind != "cpu":
        return jnp.argsort(arr,axis=axis)
    
    if axis is None:
        result_shape = arr.ravel().shape
    else:
        result_shape = arr.shape
    result_shape=jax.ShapeDtypeStruct(result_shape, np.int32)

    return jax.pure_callback(
        _run_argsort_numpy, result_shape, (arr, axis)
    )    

if jax.devices()[0].device_kind == "cpu":
    run_argsort_numpy = jax.custom_jvp(run_argsort_numpy)
    
    @run_argsort_numpy.defjvp
    def default_grad(primals, tangents):
        return run_argsort_numpy(*primals), run_argsort_numpy(*tangents)

def gen_data(mean_a, mean_b, std_a, std_b):
    a = random.normal(key, (1000,)) * std_a + mean_a
    b = random.normal(key, (2000,)) * std_b + mean_b
    return a, b


def diff_sum_top1pct(a, b, use_numpy_argsort):
    ab = jnp.concatenate((a, b))
    origin = jnp.concatenate((jnp.zeros(len(a)), jnp.ones(len(b))))
    if use_numpy_argsort:

        inds = run_argsort_numpy(-ab)

    else:
        inds = jnp.argsort(-ab)
    # return inds[0].astype(float) # to test JIT
    inds_top_1pct = inds[: len(ab) // 100]
    top1_pct = ab[inds_top_1pct]
    ori = origin[inds_top_1pct]
    top_a = top1_pct[ori == 0]
    top_b = top1_pct[ori == 1]
    return top_a.sum() - top_b.sum()

def main(mean_a: float, mean_b: float, std_a: float, std_b: float, use_numpy_argsort):
    a, b = gen_data(mean_a, mean_b, std_a, std_b)
    return diff_sum_top1pct(a, b, use_numpy_argsort=use_numpy_argsort)


grad_main = jax.grad(main, argnums=(0, 1, 2, 3))
params = 2.0, 2.0, 3.0, 4.0

print(main(*params, use_numpy_argsort=False))
result = grad_main(*params, use_numpy_argsort=False)
print([float(f) for f in result])

print(main(*params, use_numpy_argsort=True))
result = grad_main(*params, use_numpy_argsort=True)
print([float(f) for f in result])



# Test JIT
'''
main = jax.jit(main, static_argnames=("use_numpy_argsort",))
grad_main = jax.grad(main, argnums=(0, 1, 2, 3))

print(main(*params, use_numpy_argsort=False))
result = grad_main(*params, use_numpy_argsort=False)
print([float(f) for f in result])

print(main(*params, use_numpy_argsort=True))
result = grad_main(*params, use_numpy_argsort=True)
print([float(f) for f in result])
print("JIT START")
print(main(*params, use_numpy_argsort=True))
result = grad_main(*params, use_numpy_argsort=True)
print([float(f) for f in result])
print(jax.value_and_grad(main)(*params, use_numpy_argsort=True))
'''