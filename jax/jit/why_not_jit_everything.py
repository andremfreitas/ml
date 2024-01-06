import jax 
import jax.numpy as jnp

'''
 Why can’t we just JIT everything?
----------------------------------
After going through the example in 'jit.py', you might be wondering whether
 we should simply apply jax.jit to every function. To understand why
 this is not the case, and when we should/shouldn’t apply jit, let’s
 first check some cases where JIT doesn’t work.
'''

# Condition on value of x.

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

f_jit = jax.jit(f)
f_jit(10)  # Should raise an error. 

####################

# While loop conditioned on x and n.

def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

g_jit = jax.jit(g)
g_jit(10, 20)  # Should raise an error. 


'''
 The problem is that we tried to condition on the value of an input to the
 function being jitted. The reason we can’t do this is related to the fact
 mentioned above that jaxpr depends on the actual values used to trace it.


One way to deal with this problem is to rewrite the code to avoid conditionals
 on value. Another is to use special control flow operators like jax.lax.cond.
 However, sometimes that is impossible. In that case, you can consider jitting
 only part of the function. For example, if the most computationally expensive
 part of the function is inside the loop, we can JIT just that inner part (though
 make sure to check the next section on caching to avoid shooting yourself in the
 foot):
'''

# While loop conditioned on x and n with a jitted body.

@jax.jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

g_inner_jitted(10, 20)
