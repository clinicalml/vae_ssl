# Scalable Semi-Supervised Learning with Variational Autoencoders
This repo implements a series of semi-supervised variational autoencoding models in [Theano](http://deeplearning.net/software/theano/).  It uses [theanomodels](https://github.com/clinicalml/theanomodels) as an experimental framework.

# Setup
```
git clone https://github.com/clinicalml/vae_ssl
cd vae_ssl
pip install -r requirements.txt
```

### Introduction to Semi Supervised Learning with VAEs
It is often the case that labeled data is scarce, and unlabeled data is plentiful.  We denote `Y` as the label and `X` as inputs used to predict `Y`.  The goal of semi-supervised learning is to improve the performance of a supervised learning model using both labeled and unlabeled data.  The intuition is that if a model can represent both labeled and unlabeled data well, then it should have lower generalization error in the supervised setting.  In this work, we seek a probabilistic model `p` that both minimizes supervised learning error and maximizes the likelihood of observing both labeled and unlabeled data:

<a href="https://www.codecogs.com/eqnedit.php?latex=\max_{\theta_U,\theta_L}&space;\sum_{x&space;\in&space;U}&space;p(x;\theta_U)&space;&plus;&space;\sum_{x,y&space;\in&space;L}&space;p(x,y;\theta_L)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\max_{\theta_U,\theta_L}&space;\sum_{x&space;\in&space;U}&space;p(x;\theta_U)&space;&plus;&space;\sum_{x,y&space;\in&space;L}&space;p(x,y;\theta_L)" title="\max_{\theta_U,\theta_L} \sum_{x \in U} p(x;\theta_U) + \sum_{x,y \in L} p(x,y;\theta_L)" /></a>

* `X` is always observed
* `Y` is partially observed
* `U` is the unlabeled data set, consisting of only samples of `X`
* `L` is the labeled data set, consistent of samples of both `X` and `Y`
* `\lambda` is a hyperparameter for balancing the relative "strength" of the supervised and semi-supervised terms

We further assume that our data is "generated" via a latent variable generative model, like that in the figure below, `Z` the never-observed latent variable.  Using MNIST as an example, we could imagine that `Y` represents class information while `Z` represents style.

With a latent variable model, the maximum likelihood function is intractable.  So we approximate it using the variational lower bound:

Here, `p` and `q` are separate probabilistic models.  Through training, `p` and `q` learn to approximate each other while also learning to model the data in `U` and `L`.  To train a variational autoencoder, stochastic backpropagation [3,4] can be used. 

In this work we focus on supervised classification, where `Y` is discrete, however this framework could also be extended to regression.  A difficulty with the variational lower bound in classification is that as the cardinality of `Y` grows, it becomes more difficult to evaluate the sum over the states of `Y` in the lower bound.  For example, if we want to train a model on MNIST using a neural network to represent `p`, we would need to either evaluate `p` once for each of the ten states of `Y` for every mini-batch, or we would need to somehow approximate the sum.  Monte Carlo sampling of discrete variables is known to have high variance, and thus we would expect poor training performance under such a setting.  [2] developed a relaxation to this problem using the ![Gumbel-Max trick](https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/), where, during training, `Y` is relaxed to the continuous domain and sampled via calculating a softmax over Gumbel-distributed random variates.  When the temperature parameter of the softmax is tuned correctly, this approach can rival exact evaluation of the sum over `Y`.

Other approximations exist.  For example, we can introduce another latent variable, `\alpha`, which serves as a prior over `Y` in `p`.  This formulation would not require any further relaxations, though we found that in order to  


We can develop a variational autoencoding approach to solving this problem by maxim

Assume a latent variable generative model, where `Z` is always 
We assume two structurally different generative models to describe We assume a latent variable, `Z`, exists and form the variational lower bound to the objective function:

### Model Descriptions

### Training Details

### Validation Results
![](https://github.com/clinicalml/vae_ssl/blob/master/plots/multi_seed_accuracy_validation.png)

### Samples 
Samples from both $p(x)$ and $p(x,y)$ of the GumbelSoftmaxM2 model at the end of training:
![](https://github.com/clinicalml/vae_ssl/blob/master/plots/samples_GumbelSoftmaxM2.png)

### KL Plots
KL divergence at each latent node over training epochs for GumberSoftmaxM2 model:
![](https://github.com/clinicalml/vae_ssl/blob/master/plots/KL_Z_GumbelSoftmaxM2.png)

### References

[1] Kingma, Diederik P., et al. "Semi-supervised learning with deep generative models." Advances in Neural Information Processing Systems. 2014.

[2] Jang, Eric, Shixiang Gu, and Ben Poole. "Categorical reparameterization with gumbel-softmax." arXiv preprint arXiv:1611.01144 (2016). 


