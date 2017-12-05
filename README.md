# Scalable Semi-Supervised Learning with Variational Autoencoders
This repo implements a series of semi-supervised variational autoencoding models in [Theano](http://deeplearning.net/software/theano/).  It uses [theanomodels](https://github.com/clinicalml/theanomodels) as an experimental framework.

# Setup
```
git clone https://github.com/clinicalml/vae_ssl
cd vae_ssl
pip install -r requirements.txt
```

# Training
As an example, to train a Gumbel Softmax model similar to the architecture in [2], use the command:
```
python run/001_GumbelSoftmaxM2.py
```
This will train a semi-supervised variational autoencoder on MNIST using 10 random training samples from each digit class as labeled examples and all other training samples as unlabeled samples.

See `results.ipynb` for an example of how to use output results.

# Example Results
All models, except MLP, are semi-supervised variational autoencoders. The autoencoders are trained on MNIST using 10 random training samples from each digit class as labeled examples, for a total of 100 labeled samples, and all other training samples as unlabeled samples.  The MLP is trained on just the 100 labeled samples.

### Validation Accuracy 
![](https://github.com/clinicalml/vae_ssl/blob/master/plots/multi_seed_accuracy_validation.png)

### Samples 
Samples from both $p(x)$ and $p(x,y)$ of the GumbelSoftmaxM2 model at the end of training:
![](https://github.com/clinicalml/vae_ssl/blob/master/plots/samples_GumbelSoftmaxM2_shrp3.0_seed2.png)

### KL Plots
KL divergence at each latent node over training epochs for GumberSoftmaxM2 model:
![](https://github.com/clinicalml/vae_ssl/blob/master/plots/KL_Z_GumbelSoftmaxM2_shrp3.0_seed2.png)

### References

[1] Kingma, Diederik P., et al. "Semi-supervised learning with deep generative models." Advances in Neural Information Processing Systems. 2014.

[2] Jang, Eric, Shixiang Gu, and Ben Poole. "Categorical reparameterization with gumbel-softmax." arXiv preprint arXiv:1611.01144 (2016). 


