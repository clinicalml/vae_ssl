# Scalable Semi-Supervised Learning with Variational Autoencoders
This repo implements a series of semi-supervised variational autoencoding models in [Theano](http://deeplearning.net/software/theano/) using [theanomodels](https://github.com/clinicalml/theanomodels) as a framework.

[1] Kingma, Diederik P., et al. "Semi-supervised learning with deep generative models." Advances in Neural Information Processing Systems. 2014.

[2] Jang, Eric, Shixiang Gu, and Ben Poole. "Categorical reparameterization with gumbel-softmax." arXiv preprint arXiv:1611.01144 (2016). 

### Introduction to Semi Supervised Learning with VAEs
The goal of semi-supervised learning is to improve the performance of a supervised learning classifier using unlabeled data.  Semi-supervised learning is very natural with variational autoencoders.  We formulate the problem as maximizing the likelihood of observing both labeled and unlabeled data:

<a href="https://www.codecogs.com/eqnedit.php?latex=\max_{\theta_U,\theta_L}&space;\sum_{x&space;\in&space;U}&space;p(x;\theta_U)&space;&plus;&space;\sum_{x,y&space;\in&space;L}&space;p(x,y;\theta_L)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\max_{\theta_U,\theta_L}&space;\sum_{x&space;\in&space;U}&space;p(x;\theta_U)&space;&plus;&space;\sum_{x,y&space;\in&space;L}&space;p(x,y;\theta_L)" title="\max_{\theta_U,\theta_L} \sum_{x \in U} p(x;\theta_U) + \sum_{x,y \in L} p(x,y;\theta_L)" /></a>

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

