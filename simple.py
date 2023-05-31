from jax import numpy as jnp
from jax import jit, vmap
from jax.lax import cond, fori_loop

arr = jnp.array([0, 6, 7, 2, 0, 0])
arr2 = arr[arr > 0]
print(jnp.log(arr2).sum())

@jit
def inner(aa):
    return cond(aa>0, lambda: jnp.log(aa), lambda: 0.0)

@jit
def func(a):
    inner2 = vmap(inner)
    result = inner2(a)
    return result.sum()

print(func(arr))

@jit
def func(a):
    def inner2(i, s):
        return s + inner(a[i])
    result = fori_loop(0, len(a), inner2, 0)
    return result

print(func(arr))

inds = jnp.array([1,2,3])

@jit
def func(a, inds):
    return a[inds[0]] + a[inds[1]] + a[inds[2]]
print(func(arr, inds))

@jit
def func(a, inds):
    def inner3(i, s):
        return s + inner(a[inds[i]])
    result = fori_loop(0, len(inds), inner3, 0)
    return result
print(func(arr, inds))

