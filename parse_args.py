"""
Parse command line and store result in params
"""
import argparse,copy
p = argparse.ArgumentParser(description="Arguments for variational autoencoder")
parser = argparse.ArgumentParser()

#Model specification
parser.add_argument('-ph','--p_dim_hidden', action='store', default = 3, help='Hidden dimensions (in p)', type=int)
parser.add_argument('-pl','--p_layers', action='store',default = 3, help='#Layers in Generative Model', type=int)
parser.add_argument('-pzl','--z_generative_layers', action='store',default = 0, help='#Layers in hidden stack that receives [z] as input and outputs hidden activations in generative model', type=int)
parser.add_argument('-ds','--dim_stochastic', action='store',default = 5, help='Stochastic dimensions', type=int)
parser.add_argument('-ql','--q_layers', action='store',default = 3, help='#Layers in Recognition Model', type=int)
parser.add_argument('-qh','--q_dim_hidden', action='store', default = 3, help='Hidden dimensions (in q)', type=int)
parser.add_argument('-hzl','--hz_inference_layers', action='store',default = 0, help='#Layers in hidden stack that receives [h(x)] as input and outputs [hz(x)]', type=int)
parser.add_argument('-al','--alpha_inference_layers', action='store',default = 2, help='#Layers in hidden stack that receives [h(x)] as input and outputs [logbeta]', type=int)
parser.add_argument('-zl','--z_inference_layers', action='store',default = 2, help='#Layers in hidden stack that receives [alpha,hz(x)] as input and outputs [mu,logcov]', type=int)
parser.add_argument('-yl','--y_inference_layers', action='store',default = 0, help='#Layers in hidden stack that receives hz2 as input and outputs hy, (only used in GMM model)', type=int)
parser.add_argument('-bilinear','--bilinear', action='store',default = False, help='use bilinear transformation to merge alpha and h(x)', type=bool)
parser.add_argument('-nc','--nclasses', action='store', default = 10, help='# classes in labels', type=int)
parser.add_argument('-nlpc','--nlabelsperclass', action='store', default = 10, help='# labeled examples per label class', type=int)
parser.add_argument('-betaprior','--betaprior', action='store', default = 0.2, help='dirichlet prior', type=float)
parser.add_argument('-finalbeta','--finalbeta', action='store', default = 0.2, help='dirichlet prior final', type=float)
parser.add_argument('-annealBP','--annealBP', action='store', default = 1., help='dirichlet prior annealing', type=float)
parser.add_argument('-betamax','--betamax', action='store', default = 10.0, help='setting for MixDirPrior model', type=float)
parser.add_argument('-logp_S','--logp_S', action='store', default = 100, help='setting for MixDirPrior model', type=int)
parser.add_argument('-sharpening','--sharpening', action='store', default = 1., help='softmax sharpening', type=float)
parser.add_argument('-annealSharpening','--annealSharpening', action='store', default = 1., help='anneal sharpening', type=float)
parser.add_argument('-no_softmax','--no_softmax', action='store', default = False, help='Dont use softmax for alpha in VAE, but still use it in classifier', type=bool)
parser.add_argument('-kllg','--KL_loggamma_coef', action='store', default = 1.0, help='Coefficient multiplier to the KL of the loggamma RVs', type=float)
parser.add_argument('-mbw','--maxBetaWeight', action='store', default = 0., help='Coefficient multiplier to the distance between max logbeta and second max logbeta in U', type=float)
parser.add_argument('-mbwxy','--maxBetaWeightXY', action='store', default = 0., help='Coefficient multiplier to the distance between max logbeta and second max logbeta in L', type=float)
parser.add_argument('-cw','--classifier_weight', action='store', default = 128.0, help='Coefficient multiplier to the classification loss', type=float)
parser.add_argument('-bxy','--boundXY_weight', action='store', default = 1.0, help='Coefficient multiplier to the variational bound of L', type=float)
parser.add_argument('-pxyd','--logpxy_discrete', action='store', default = False, help='Discrete model for logpxy', type=bool)

parser.add_argument('-idrop','--input_dropout', action='store',default = 0.25, help='Dropout at input',type=float)
parser.add_argument('-betadrop','--dropout_logbeta', action='store',default = 0., help='Dropout on logbeta',type=float)
parser.add_argument('-hxdrop','--dropout_hx', action='store',default = 0., help='Dropout on into z',type=float)
parser.add_argument('-nl','--nonlinearity', action='store',default = 'maxout', help='Nonlinarity',type=str, choices=['relu','tanh','softplus','maxout','maxout2'])
parser.add_argument('-ischeme','--init_scheme', action='store',default = 'uniform', help='Type of initialization for weights', type=str, choices=['uniform','normal','xavier','he'])
parser.add_argument('-mstride','--maxout_stride', action='store',default = 4, help='Stride for maxout',type=int)
parser.add_argument('-lky','--leaky_param', action='store',default =0., help='Leaky ReLU parameter',type=float)
parser.add_argument('-iw','--init_weight', action='store',default = 0.1, help='Range to initialize weights during learning',type=float)

parser.add_argument('-bn','--batchnorm', action='store',default = False, help='Batch Normalization',type=bool)
parser.add_argument('-mbn','--modifiedBatchNorm', action='store',default = False, help='Modified Batch Normalization',type=bool)
parser.add_argument('-static_mBN','--static_mBN', action='store',default = False, help='Static Modified Batch Normalization',type=bool)
parser.add_argument('-sbn','--separateBNrunningstats', action='store',default = False, help='Separate Batch Normalization running stats between supervised and unsupervised models',type=bool)
parser.add_argument('-ln','--layernorm', action='store',default = False, help='Layer Normalization',type=bool)
parser.add_argument('-pn','--p_normlayers', action='store',default = False, help='Allow normalization layers in Generative Model', type=bool)

parser.add_argument('-model','--model',action='store',default='LogGamma',help='choose type of graphical model',choices=['LogGamma','ExactM2','ApproxM2','MixDirPrior','LogisticNormal','LogisticNormalMP','DirichletMixture','Dirichlet2','LogGammaMixture','LogGammaLatentMixture','LogGammaReverse','DirichletSeparate','LogisticNormal_fp','GM','GM2','LNprd','GumbelSoftmaxM2','STGumbelSoftmaxM2','LogisticNormalM2','STLogisticNormalM2','GumbelSoftmax','STGumbelSoftmax','LogisticNormal_warped','STLogisticNormal_warped'])
parser.add_argument('-lnmp','--LogitNormalMP',action='store',default=3.,help='LogitNormalMP coefficient',type=float)
parser.add_argument('-lpr','--learn_prior',action='store',default=False,help='LogGammaMixture learn prior',type=bool)
parser.add_argument('-lpo','--learn_posterior',action='store',default=False,help='learn posterior (model-specific implementations)',type=bool)
parser.add_argument('-pv','--posterior_val',action='store',default=-6.,help='scalar value used in adjusting the posterior of the model conditioned on the class label',type=float)

#Optimization
parser.add_argument('-dset','--dataset', action='store',default = '', help='Dataset', type=str)
parser.add_argument('-lr','--lr', action='store',default = 5e-4, help='Learning rate', type=float)
parser.add_argument('-lrd','--lr_decay', action='store',default = False, help='Learning rate decay', type=bool)
parser.add_argument('-opt','--optimizer', action='store',default = 'adam', help='Optimizer',choices=['adam','rmsprop'])
parser.add_argument('-bs','--batchsize', action='store',default = 100, help='Batch Size',type=int)
parser.add_argument('-gn','--grad_norm', action='store',default = 2.5, help='max grad norm per 1000 parameters',type=float)
parser.add_argument('-dg','--divide_grad', action='store',default = False, help='Rescale grad to batch size',type=float)
parser.add_argument('-aklz','--annealKL_Z', action='store',default = 1, help='# iterations to anneal KL terms in variational bound (i.e. warmup)',type=int)
parser.add_argument('-akla','--annealKL_alpha', action='store',default = 1, help='# iterations to anneal KL terms in variational bound (i.e. warmup)',type=int)
parser.add_argument('-ancw','--annealCW', action='store',default = 50000, help='# iterations to anneal classification weight',type=float)
parser.add_argument('-anbd','--annealBound', action='store',default = 1, help='# iterations to anneal variational bound',type=float)
parser.add_argument('-ns','--num_samples', action='store', default = 1, help='number of random samples per instance during training', type=int)
parser.add_argument('-maxiters','--maxiters', action='store', default = None, help='max iterations per epoch; this is useful for debugging', type=int)

#Recreation of old torch model
parser.add_argument('-negKL','--negKL', action='store',default = False, help='negative KL trick',type=bool)


#Setup 
parser.add_argument('-viz','--visualize_model', action='store',default = False,help='Visualize Model',type=bool)
parser.add_argument('-uid','--unique_id', action='store',default = None,help='Unique Identifier',type=str)
parser.add_argument('-seed','--seed', action='store',default = 1, help='Random Seed',type=int)
parser.add_argument('-dir','--savedir', action='store',default = './chkpt', help='Prefix for savedir',type=str)
parser.add_argument('-ep','--epochs', action='store',default = 500, help='MaxEpochs',type=int)
parser.add_argument('-reload','--reloadDir', action='store',default = None, help='Directory to reload model from',type=str)
parser.add_argument('-config','--configFile', action='store',default = 'config.json', help='Filename used to save config file (will be saved to json format)',type=str)
parser.add_argument('-sfreq','--savefreq', action='store',default = 100, help='Frequency of saving',type=int)
parser.add_argument('-efreq','--evalfreq', action='store',default = 10, help='Frequency of evaluation on validation set',type=int)
parser.add_argument('-savemodel','--savemodel', action='store',default = False, help='Save model params (requires a lot more storage)',type=bool)
parser.add_argument('-infm','--inference_model', action='store',default = 'single', help='Inference Model',type=str, choices=['single'])
parser.add_argument('-debug','--debug', action='store',default = False, help='Debug Mode',type=bool)
parser.add_argument('-tp','--track_params', action='store',default = False, help='Track hard-coded list of parameters',type=bool)

#Regularization
parser.add_argument('-reg','--reg_type', action='store',default = 'l2', help='Type of regularization',type=str,choices=['l1','l2'])
parser.add_argument('-rv','--reg_value', action='store',default = 0.1, help='Amount of regularization',type=float)
parser.add_argument('-rspec','--reg_spec', action='store',default = '_', help='String to match parameters (Default is generative model)',type=str)
params = vars(parser.parse_args())


if __name__=='__main__':
	import json
	with open('test_params.json','w') as f:
		f.write(json.dumps(params,sort_keys=True,indent=2,separators=(',',':')))
	with open('test_params.json','r') as f:
		x = json.loads(f.read())
	for k,v in sorted(x.iteritems()):
		print '%s:' % k, v
