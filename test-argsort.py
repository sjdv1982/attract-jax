import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import host_callback as hcb

key = random.PRNGKey(42)


def gen_data(mean_a, mean_b, std_a, std_b):
    a = random.normal(key, (1000,)) * std_a + mean_a
    b = random.normal(key, (2000,)) * std_b + mean_b
    return a, b


def diff_sum_top1pct(a, b, use_numpy_argsort):
    ab = jnp.concatenate((a, b))
    origin = jnp.concatenate((jnp.zeros(len(a)), jnp.ones(len(b))))
    if use_numpy_argsort:
        ########################################
        # Doesn't work: use private API to override identity=False
        #inds = hcb._call(
        #    np.argsort,
        #    -ab,
        #    result_shape=jax.ShapeDtypeStruct(ab.shape, np.int32),
        #    identity=True,
        #    call_with_device=False,
        #    device_index=0,
        #)
        ########################################

        ########################################
        # Requires to comment out in jax/experimental/host_callback.py the following lines:
        # 1402: if not params["identity"]:
        # 1403:   raise NotImplementedError("JVP rule is implemented only for id_tap, not for call.")

        inds = hcb.call(
            np.argsort, -ab, result_shape=jax.ShapeDtypeStruct(ab.shape, np.int32)
        )
        ########################################

    else:
        inds = jnp.argsort(-ab)
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

print(main(*params, use_numpy_argsort=False))

result = grad_main(*params, use_numpy_argsort=True)
print([float(f) for f in result])
