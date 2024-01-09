'''
 How JAX transforms work
 -----------------------

In the previous section, we discussed that JAX allows us to transform Python functions,
This is done by first converting the Python function into a simple intermdeaited languiage
called *jaxpr*. The transformations then work on the jaxpr representation.

We can show a representation of the jaxpr of a function by using 'jax.make_jaxpr'.
'''

import jax 
import jax.numpy as jnp

global_list = []

def log2(x):
 global_list.append(x)
 ln_x = jnp.log(x)
 ln_2 = jnp.log(2.0)
 return ln_x / ln_2

print(jax.make_jaxpr(log2)(3.0))

# Here we can see two things: the output in jaxpr form. And also that in there 
# there is nothing about the appending being done in the function, because this is
# a side effect and not functionally pure. 

# Note: the Python print() function is not pure: the text output is a side-effect
# of the function. Therefore, any print() calls will only happen during tracing,
# and will not appear in the jaxpr:

def log2_with_printing(x):
  print('printed x:', x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2_with_printing)(3.))


'''
See how the printed x is a Traced object? That’s the JAX internals at work.

The fact that the Python code runs at least once is strictly an implementation detail,
and so shouldn’t be relied upon. However, it’s useful to understand as you can use it
when debugging to print out intermediate values of a computation.

A key thing to understand is that jaxpr captures the function as executed on the
parameters given to it. For example, if we have a conditional, jaxpr will only
know about the branch we take:
'''

def log_if_rank_2(x):
 if x.ndim == 2: 
   ln_x = jnp.log(x)
   ln_2 = jnp.log(2.0)
   return ln_x / ln_2
 else:
   return x

print(jax.make_jaxpr(log_if_rank_2)(jnp.array([1,2,3])))
