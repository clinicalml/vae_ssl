# Models

## Baseline MLP

This is a fully-connected work that shares the same architecture as that of `q(y|x)` in all of the models described below, except the LogisticNormal model, in which q(y|x) is estimated with Monte Carlo sampling.

## Models based on the M2 model in [1]

![](pngs/M2_diagram.png)
![](pngs/M2_variational_bound_U.png)
![](pngs/M2_variational_bound_L.png)
![](pngs/M2_variational_objective.png)

### ExactM2

The ExactM2 model is "exact" in the sense that, during training, it evaluates every element in the summation over the states of `Y` in the objective function.

```
p(y) ~ Categorical
p(z) ~ Normal
p(x_i|z,y) ~ Bernoulli()
```

### ApproxM2

The ApproxM2 model is "approximate" in the sense that, during training, it estimates the expecation over `q(y|x)` in the objective function via Monte Carlo sampling of the categorical distribution over the states of `Y`.  This is not actually a good sampling strategy, since the gradient of this approximation while have high variance and lead to poor performance. Black box variational inference [3] is used to estimate the gradient through the expecation over `Y`.

### GumbelSoftmaxM2 [2]

This is the model described in [2], which relax the sampling of `q(y|x)` in the variational bound of `logp(x)` to a Gumbel-softmax distribution, which lies on a continuous-domain simplex, and leverage the [Gumbel-Max trick](https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/) to show that this distribution converges to a categorical in the limit as the temperature parameter of the softmax goes to inifinity.  Note that `KL[q(y|x)||p(y)]` is still evaluated using the categorical distribution `q(y|x)`, not the relaxation.  The performance of this model is sensitive to the tuning of the temperature parameter.  See more about this at [](http://blog.evjang.com/2016/11/tutorial-categorical-variational.html).

### LogisticNormalM2

This copies the model framework and training protocol of the GumbelSoftmaxM2, except that a logistic-normal, i.e. softmax over a normal distribution, is used in place of the Gumbel-softmax during sampling of `q(y|x)`.

## Another set of models 

### LogGamma

### LogisticNormal



