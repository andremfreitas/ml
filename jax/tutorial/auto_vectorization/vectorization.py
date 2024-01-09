'''
Automatic Vectorization in JAX
------------------------------

We previously saw the JIT compilation via the function 'jax.jit'.
Now, we look into another of JAX transforms: vectorization via
'jax.vmap'

'''

# Consider the following simple code that computes the convolution of 
# two 1D arrays:

import jax
import jax.numpy as jnp

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

def convolve(x,y):
 output = []
 for i in range(1, len(x) - 1):
  output.append(jnp.dot(x[i-1:i+2], w))
 return jnp.array(output)

convolve(x,w)

# Let's suppose we'd like to apply this function to a batch of weights
# w and a batch of vectors x

xs = jnp.stack([x,x])
ws = jnp.stack([w,w])

# The most naive option would be to simply loop over the batch in Python

def manually_batched_convolve(xs, ws):
 output = []
 for i in range(xs.shape[0]):
  output.append(convolve(xs[i], ws[i]))
  return jnp.stack(output)

manually_batched_convolve(xs,ws)

# While this produces the correct results, it is not efficient

# We could manually vectorize computation, but this would require
# changing the indicing, inputs and so on... Lets do it here

def manually_vectorized_convolve(xs,ws):
 output = []
 for i in range(1, xs.shape[-1] -1):
  output.append(jnp.sum(xs[:, i-1:i+2] * ws, axis=1))
  return jnp.stack(output, axis=1)

manually_vectorized_convolve(xs,ws)

# While such implementation is possible, it is error prone and costs time...
# JAX offers an alternative

# In JAX, the 
