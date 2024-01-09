import jax 
import jax.numpy as jnp
import timeit 

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)

def benchmark_selu():
    selu(x).block_until_ready()

# Measure the execution time using timeit
time_taken = timeit.timeit(benchmark_selu, number=100)
print(f"Time taken: {time_taken} seconds")

'''
 The code above is sending one operation at a time to the accelerator. 
This limits the ability of the XLA compiler to optimize our functions.

Naturally, what we want to do is give the XLA compiler as much code as
 possible, so it can fully optimize it. For this purpose, JAX provides
 the jax.jit transformation, which will JIT compile a JAX-compatible function.
 The example below shows how to use JIT to speed up the previous function.
'''

selu_jit = jax.jit(selu)

def jitted_selu():
  selu_jit(x).block_until_ready() # this warm up is just to not count the compilation time during the timing (to make it fairer)

time_taken_jit = timeit.timeit(jitted_selu, number=100)
print(f"Time taken: {time_taken_jit} seconds")

