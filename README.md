# Machine learning exercises/projects

Archive for exercises/projects performed as I learn ML/DL.
*  Curve fitting

## Curve fitting

We generate some noisy data:

```math

y(x) = 0.1 x \cos(x) + 0.1\epsilon \,\,,

```

where $\epsilon$ is normally distributed (with $\mu$ = 0 and $\sigma$ = 1). We try to fit this using a deep NN. After some hyparameter space search, a NN with two hidden layers of 32 neurons each was designed. The NN has one input node and one output node ($\mathbb{R}$ $\to$ $\mathbb{R}$). In terms of training 100 epochs are performed. There are 1000 data points for training and 100 for validation. 

**Libraries needed**: numPy, tensorflow, matplotlib.

Code:
* `fit.py` is the *begginer* version of the implementation curve fitting using tensorflow. Less user control and a more black-box like approach.
* `expert_fit.py` is the more *advanced* version, where we use `tf.GradientTape()`, define our model as a class that inherits from `tf.keras.Model`, among other things.


## Learning the equations of motion

Here, the goal is to use the NN as a predictor. Based on the current state, the goal of the NN is to predict the following states (evolve in time). The problem chosen is the evolution of a trajectory in time in a 2D physical space. The NN has as input at time $t$ [x, y, vx, vy] and needs to predict [x, y, vx, vy] $t$ + $\delta t$ ($\mathbb{R}^4$ $\to$ $\mathbb{R}^4$) . For this, trajectory data is generated using random ICs. The NN learns from this data. The loss function used is the mean of the rollout of the squared pointwise error. The performance of the NN is tested on unseen ICs.


