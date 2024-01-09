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

# In JAX, the 'jax.vmap' transformation is desgined to generate
# such a vectorized implementation of a function automatically:

auto_batch_convolve = jax.vmap(convolve)

auto_batch_convolve(xs,ws)

'''
It does this by tracing the function similarly to jax.jit, and 
automatically adding batch axes at the beginning of each input.

If the batch dimension is not the first, you may use the in_axes
and out_axes arguments to specify the location of the batch dimension
in inputs and outputs. These may be an integer if the batch axis is
the same for all inputs and outputs, or lists, otherwise.
'''

auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)

auto_batch_convolve_v2(xst, wst)


'''

Combining transformations
------------------------- 

As with all JAX transformations, 'jax.jit' and 'jax.vmap' are composable.
Which means we can wrap a jitted function with a vmap and the other way around.

'''

jitted_batch_convolve = jax.jit(auto_batch_convolve)

jitted_batch_convolve(xs, ws)
