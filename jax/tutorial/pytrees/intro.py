'''
What is a pytree ? 
-----------------

A pytree is a container of leaf elements and/or more pytrees. Containers
include lists, tuples, and dicts. A leaf element is anything that's not a pytree,
e.g. an array. In other words, a pytree is just a possibly-nested standard 
or user-registered Python container. If nested, note that the container
types do not need to match. 
'''

# First example of a pytree

import jax
import jax.numpy as jnp

example_trees = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]

# Let's see how many leaves they have:
for pytree in example_trees:
  leaves = jax.tree_util.tree_leaves(pytree)
  print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")

'''
Common pytree functions:
-----------------------

Perhaps the most commonly used pytree function is 'jax.tree_map'. It works
analogously to the python's native function 'map', but on pytrees.
'''

list_of_lists = [
 [1,2,3],
 [1,2],
 [1,2,3,4]
]

print(jax.tree_map(lambda x: x*2, list_of_lists))
