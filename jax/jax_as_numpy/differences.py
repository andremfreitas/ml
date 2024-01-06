'''
The jax.numpy API closely follows that of NumPy. 
However, there are some important differences

The most important difference, and in some sense the root of all the rest, is that JAX is designed to be functional, as in functional programming. 
The reason behind this is that the kinds of program transformations that JAX enables
are much more feasible in functional-style programs.

The important feature of functional programming to grok when working with JAX is 
very simple: don’t write code with side-effects.

A side-effect is any effect of a function that doesn’t appear in its output.
One example is modifying an array in place:
'''

import numpy as np

x = np.array([1, 2, 3])

def in_place_modify(x):
 x[0] = 123
 return None

in_place_modify(x)
print(x)

# The side-effectful function modifies its argument, but returns a completely unrelated value.
# The modification is a side-effect.

# In NumPy the code runs, but not in JAX
import jax
import jax.numpy as jnp

#in_place_modify(jnp.array(x))   # -- produces an error

# The correct way to do this:

def jax_in_place_modify(x):
 return x.at[0].set(123)

y = jnp.array([1,2,3])

print(jax_in_place_modify(y))

# We can see  that actually the old array waas untouched. So there is no side-effect.
print(y)

# Side-effect-free code is sometimes called *functionally pure* or just *pure*.

'''
Isn’t the pure version less efficient? Strictly, yes; we are creating a new array.
However, as we will explain in the next guide, JAX computations are often compiled
before being run using another program transformation, jax.jit. If we don’t use the
old array after modifying it ‘in place’ using indexed update operators, the compiler
can recognise that it can in fact compile to an in-place modify, resulting in efficient
code in the end.

Of course, it’s possible to mix side-effectful Python code and functionally pure JAX
code, and we will touch on this more later. As you get more familiar with JAX, you
will learn how and when this can work. As a rule of thumb, however, any functions
intended to be transformed by JAX should avoid side-effects, and the JAX primitives
themselves will try to help you do that.
'''
