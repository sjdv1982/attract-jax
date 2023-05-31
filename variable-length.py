import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import cond, fori_loop
import time

ELE = 1000000
MAXITER = 4
CONVOLUTION = 200

### CONVOLUTION = 2 ###
### ELE = 10000 ###

np.random.seed(7)
list_lengths = np.random.choice([0,1,2,3,4], ELE, p=[0.77,0.12,0.08,0.02,0.01] )
var_lists = [[10*i+ii for ii in range(l)] for i,l in enumerate(list_lengths) ]
const_lists = [[10*i+ii for ii in range(4)] for i in range(ELE) ]

### print(var_lists[:16])

var_data = np.zeros(4 * ELE, np.uint32)
var_data0 = np.concatenate(var_lists)
var_data[:len(var_data0)] = var_data0
### print(var_data[:7])
var_indices = np.zeros((ELE, 2),np.uint32)
var_indices[:, 0] = list_lengths
var_indices[:, 1][1:] = np.cumsum(list_lengths)[:-1]
var_indices[:, 1][list_lengths==0] = 0
### print(var_indices[:16].tolist())

const_data = np.concatenate(const_lists).astype(np.uint32)
### print(const_data[:7])
const_indices = np.zeros((ELE, 2),np.uint32)
const_indices[:, 0] = 4
const_indices[:, 1][1:] = np.cumsum(np.repeat(4, ELE))[:-1]
### print(const_indices[:16].tolist())

t = time.time()
dummy_result = np.array([sum(lis, 0.0) for lis in var_lists], np.float32) ###
print("Dummy variable lists", time.time() - t)
t = time.time()
dummy_result = np.array([sum(lis, 0.0) for lis in const_lists], np.float32) ###
print("Dummy constant lists", time.time() - t)

noise = jnp.array(np.random.random(100000))
noise0 = np.array(noise)

@jit
def f(v):
    result = v
    for _ in range(CONVOLUTION):
        result = jnp.log(jnp.abs(result))**2
        k = (100000*(result - jnp.floor(result) )).astype(int)
        result = jnp.arcsin(noise[k])
    return result

@jit
def ff(length_start, data):
    length, start = length_start
    def inner(i, val):
        return cond(i<length, lambda: f(data[start+i]), lambda: 0.0) + val
    return fori_loop(0, MAXITER, inner, 0.0)

main = vmap(ff,(0, None))

### print(f(0.7))
### print(f(0.7))
result = main(var_indices, var_data)
jax.block_until_ready(result)

print("Start", var_indices.shape, var_data.shape, const_indices.shape, const_data.shape)
t = time.time()
result = main(var_indices, var_data)
jax.block_until_ready(result)
print("Variable lists", time.time() - t)

t = time.time()
result = main(const_indices, const_data)
jax.block_until_ready(result)
print("Constant lists", time.time() - t)

t = time.time()
result = main(var_indices, var_data)
jax.block_until_ready(result)
print("Variable lists", time.time() - t)

t = time.time()
result = main(const_indices, const_data)
jax.block_until_ready(result)
print("Constant lists", time.time() - t)

t = time.time()
result = main(var_indices, var_data)
jax.block_until_ready(result)
print("Variable lists", time.time() - t)

print("Reference variable lists")
t = time.time()
refe_result0 = [[f(v) for v in lis] for lis in var_lists] ###
refe_result = np.array([sum(lis, 0.0) for lis in refe_result0], np.float32) ###
print("Reference variable lists", time.time() - t)

print(len(result), np.max(np.abs(result-refe_result))) ###


t = time.time()
refe_result = np.array([sum([float(jax.block_until_ready(f(v))) for v in lis], 0.0) for lis in var_lists], np.float32) ###
print("Reference variable lists, blocking",time.time() - t)

print(len(result), np.max(np.abs(result-refe_result))) ###

t = time.time()
result = main(const_indices, const_data)
jax.block_until_ready(result)
print("Constant lists", time.time() - t)

print("Reference constant lists")
t = time.time()
refe_result0 = [[f(v) for v in lis] for lis in const_lists] ###
refe_result = np.array([sum(lis, 0.0) for lis in refe_result0], np.float32) ###
print("Reference constant lists", time.time() - t)

print(len(result), np.max(np.abs(result-refe_result))) ###

t = time.time()
refe_result = np.array([sum([float(jax.block_until_ready(f(v))) for v in lis], 0.0) for lis in const_lists], np.float32) ###
print("Reference constant lists, blocking", time.time() - t)

print(len(result), np.max(np.abs(result-refe_result))) ###
