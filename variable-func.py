import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import cond, fori_loop
import time
from functools import partial

# Section 1: parallel dispatch (lack thereof)

@partial(jit, static_argnums=(0,))
def f(x):
    def inner(i, value):
        k = jnp.abs(jnp.log(jnp.abs(i+1)))
        kk = k - jnp.floor(k)
        return value + jnp.arcsin(kk)
    return fori_loop(0, 1000000 * x, inner, 0)

X = 50
jax.block_until_ready(f(X))

t = time.time()
result = sum(vmap(f)(jnp.repeat(X, 10)))
#print(result)
print(time.time() - t) # ~15 seconds, on 1 core

t = time.time()
result = []
for n in range(10):
    result.append(jax.block_until_ready(f(X)))
result = sum(result)
#print(result)
print(time.time() - t) # ~15 seconds, on 1 core

t = time.time()
result = []
for n in range(10):
    result.append(f(X))
result = sum(result)
print(time.time() - t) # ~15 seconds, on 1 core !!!

# Section 2: cond is inefficient
    
def f0(x):
    mat = np.arange(1000000).reshape(1000,1000) + x
    return mat.dot(mat).sum()

# Numpy: ~8 secs on 1 core, 10 iterations
t = time.time()
for n in range(10):
    f0(n)
print(time.time() - t)

@jit
def f(x):
    mat = jnp.arange(1000000).reshape(1000,1000) + x
    return mat.dot(mat).sum()

ff = jit(vmap(f))


# JAX CPU: ~10 secs on 12 cores, 200 iterations (same speed per core)
t = time.time()
result = []
for n in range(200):
    result.append(f(n))
sum(result)
print(time.time() - t)

@partial(jit, static_argnums=(0,1))
def main(max_iter, real_iter):
    def inner(i):        
        # No difference:
        # branches = [f, lambda i: 0]
        # return jax.lax.switch((i<real_iter).astype, branches, i)        
        return cond(i<real_iter, lambda: f(i), lambda: 0)
    arr = jnp.arange(max_iter)
    return vmap(inner)(arr)

def run(max_iter, real_iter):
    jax.block_until_ready(main(max_iter, real_iter))
    t = time.time()
    result  = main(max_iter, real_iter)
    jax.block_until_ready(result)
    print(sum(result))
    return time.time() - t

def run2(max_iter, real_iter):
    irange = jnp.arange(0, real_iter, dtype=int)
    jax.block_until_ready(ff(irange))
    
    t = time.time()
    result0 = jnp.zeros(real_iter, np.int32)
    result1 = ff(irange)
    result = jnp.empty(max_iter, np.int32)
    result = result.at[:real_iter].set(result1)
    result = result.at[real_iter:max_iter].set(result0)
    print(sum(result))
    return time.time() - t

print(run(100, 100))  # ~5 seconds
print(run(200, 100))  # ~10 seconds
print(run(200, 200))  # ~10 seconds

print()

print(run2(100, 100))  # ~5 seconds
print(run2(200, 100))  # ~5 seconds
print(run2(200, 200))  # ~10 seconds


# GPU (T4)
print(run(500, 500))  # ~1.7 seconds
print(run(1000, 500))  # ~3.5 seconds
print(run(1000, 1000))  # ~3.5 seconds
# 15x speed-up over 12 CPUS (180x over 1 CPU), but no difference: 
#   scales with max_iter, not real_iter!