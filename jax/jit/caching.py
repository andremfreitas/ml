'''
Caching
-------

It’s important to understand the caching behaviour of jax.jit.

Suppose I define f = jax.jit(g). When I first invoke f, it will get compiled,
and the resulting XLA code will get cached. Subsequent calls of f will reuse
the cached code. This is how jax.jit makes up for the up-front cost of compilation.

If I specify static_argnums, then the cached code will be used only for the same
values of arguments labelled as static. If any of them change, recompilation occurs. If there are many values, then your program might spend more time compiling than it would have executing ops one-by-one.

Avoid calling jax.jit inside loops. For most cases, JAX will be able to use the
compiled, cached function in subsequent calls to jax.jit. However, because the
cache relies on the hash of the function, it becomes problematic when equivalent
functions are redefined. This will cause unnecessary compilation each time in the
loop.
'''

import jax
import jax.numpy as jnp
from functools import partial
import timeit

from functools import partial

def unjitted_loop_body(prev_i):
  return prev_i + 1

# The following functions use JAX's jit compilation

# This function uses partial, which creates a new function each time, affecting caching.
def g_inner_jitted_partial(x, n):
  i = 0
  while i < n:
    # Don't do this! each time the partial returns
    # a function with different hash
    i = jax.jit(partial(unjitted_loop_body))(i)
  return x + i

# This function uses lambda, which also creates new function objects, affecting caching.
def g_inner_jitted_lambda(x, n):
  i = 0
  while i < n:
    # Don't do this!, lambda will also return
    # a function with a different hash
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
  return x + i

# This function directly uses jax.jit(unjitted_loop_body), allowing JAX to find and use the cached, compiled function.
def g_inner_jitted_normal(x, n):
  i = 0
  while i < n:
    # this is OK, since JAX can find the
    # cached, compiled function
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

# Function to measure the execution time using timeit
def measure_time(func):
    return timeit.timeit(lambda: func(10, 20).block_until_ready(), number=100)

# Measure and print the execution time for each function
print("jit called in a loop with partials:")
time_taken_partial = measure_time(g_inner_jitted_partial)
print(f"Time taken: {time_taken_partial} seconds")

print("jit called in a loop with lambdas:")
time_taken_lambda = measure_time(g_inner_jitted_lambda)
print(f"Time taken: {time_taken_lambda} seconds")

print("jit called in a loop with caching:")
time_taken_normal = measure_time(g_inner_jitted_normal)
print(f"Time taken: {time_taken_normal} seconds")
