# Machine learning small tasks for learning

Archive for exercises/projects performed as I learn ML/DL.
*  Curve fitting

## Curve fitting

We generate some noisy data:

```math

y(x) = 0.1 x \cos(x) + 0.1\epsilon \,\,,

```

where $\epsilon$ is normally distributed (with $\mu$ = 0 and $\sigma$ = 1). We try to fit this using a deep NN. After some hyparameter space search, a NN with two hidden layers of 32 neurons each was designed. The NN has one input node and one output node ($\mathbb{R}$ $\to$ $\mathbb{R}$). In terms of training 100 epochs are performed. There are 1000 data points for training and 100 for validation. 

Libraries needed: numPy, tensorflow, matplotlib.


