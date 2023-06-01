import jax
from jax import jit
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, sparsify
import time

data = jnp.array([1., 3., 5.])
indices = jnp.array([[0, 0],
                     [1, 1],
                     [2, 2]])
mat1 = BCOO((data, indices), shape=(3, 3))

data = jnp.array([8., 7., 3.])
indices = jnp.array([[0, 1],
                     [1, 0],
                     [2, 0]])
mat2 = BCOO((data, indices), shape=(3, 3))
print(mat2.todense())

@jit
@sparsify
def add(mat1, mat2):
    return mat1 + mat2

mat3 = add(mat1, mat2)
print(mat3)
print(mat3.todense())


def matmult0(mat1, mat2):
    return mat1.dot(mat2).sum() + mat2.dot(mat1).sum()

import numpy as np
np.random.seed(0)

for p in (0.000001, 0.001):#, 0.2, 0.5):
    mask1 = np.random.choice([False, True], p=[1-p, p], size=(2000,2000))
    mask2 = np.random.choice([False, True], p=[1-p, p], size=(2000,2000))

    print(p)
    
    mmat1 = jnp.array(mask1,dtype=jnp.float32)
    mmat2 = jnp.array(mask2,dtype=jnp.float32)
    smat1 = BCOO.fromdense(mmat1)
    smat2 = BCOO.fromdense(mmat2)

    t = time.time()
    print(matmult0(mmat1, mmat2))
    print("No jit, time:", time.time() - t)

    matmult = jit(matmult0)
    jax.block_until_ready(matmult(mmat1, mmat2))
    t = time.time()
    print(matmult(mmat1, mmat2))
    print("Jit, time:", time.time() - t)

    smatmult = sparsify(matmult0)
    jax.block_until_ready(smatmult(smat1, smat2))
    t = time.time()
    print(smatmult(smat1, smat2))
    print("Sparsify, time:", time.time() - t)
    
    smatmult = jit(sparsify(matmult0))
    jax.block_until_ready(smatmult(smat1, smat2))
    t = time.time()
    print(smatmult(smat1, smat2))
    print("Sparsify+jit, time:", time.time() - t)
    print()