import argparse
from cmath import log
from math import expm1
import os
import time as tm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as ft
from torch.distributions import constraints
from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs
from torch.nn.modules.linear import Linear
from utils.custom_mlp import MLP, Exp
from utils.scdata_cached import setup_data_loader, SingleCellCached

import pyro
import pyro.distributions as dist
from pyro.contrib.examples.util import print_and_log
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam, ExponentialLR

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef


class scClassifier(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on single cell datasets
    :param output_size:  size of the tensor representing the class label
    :param input_size: size of the tensor representing the cell
    :param z_dim: size of the tensor representing the latent random variable z
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param use_cude: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """

    def __init__(self,
                 output_size = 10,
                 input_size = 2000,
                 z_dim = 50,
                 hidden_layers = (500,),
                 label_names = None,
                 config_enum = None,
                 use_cuda = False,
                 aux_loss_multiplier = None,
                 use_mask = True,
                 dist_model = 'dmm',
                 mask_alpha = 1,
                 mask_beta = 10,
                 use_siamese = False,
                 use_zeroinflate = False,
                 gate_prior = 0.7,
                 delta = 0.5,
                 loss_func = 'multinomial',
                 dirimulti_mass = 1,
                 label_type = 'categorical',
    ):
        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.label_names = label_names
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier
        self.use_mask = use_mask
        self.mask_alpha = mask_alpha
        self.mask_beta = mask_beta
        self.dist_model = dist_model
        self.use_siamese = use_siamese
        self.use_zeroinflate = use_zeroinflate
        self.delta = delta
        self.loss_func = loss_func
        self.dirimulti_mass = dirimulti_mass
        self.label_type = label_type

        if gate_prior < 1e-5:
            gate_prior = 1e-5
        elif gate_prior==1:
            gate_prior=1-1e-5
        self.gate_prior = np.log(gate_prior) - np.log(1-gate_prior)

        # define and instantiate the neural networks representing
        # the parameters of various distributions in the model
        self.setup_networks()

    def setup_networks(self):
        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers

        # define the neural networks used later in the model and the guide.
        self.encoder_zy = MLP(
            [self.input_size] + hidden_sizes + [[z_dim, z_dim]],
            activation =  nn.Softplus,
            output_activation = [None, Exp],
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )

        if self.label_type == 'categorical':
            self.encoder_zy_y = MLP(
                [self.z_dim] + hidden_sizes + [self.output_size],
                activation = nn.Softplus,
                output_activation = None,
                allow_broadcast = self.allow_broadcast,
                use_cuda = self.use_cuda,
            )
        elif self.label_type == 'compositional':
            self.encoder_zy_y = MLP(
                [self.z_dim] + hidden_sizes + [self.output_size],
                activation = nn.Softplus,
                output_activation = nn.Softplus,
                allow_broadcast = self.allow_broadcast,
                use_cuda = self.use_cuda,
            )
        elif self.label_type in ['rate','onehot']:
            self.encoder_zy_y = MLP(
                [self.z_dim] + hidden_sizes + [[self.output_size, self.output_size]],
                activation = nn.Softplus,
                output_activation = [nn.Softplus, nn.Softplus],
                allow_broadcast = self.allow_broadcast,
                use_cuda = self.use_cuda,
            )
        elif self.label_type == 'real':
            self.encoder_zy_y = MLP(
                [self.z_dim] + hidden_sizes + [[self.output_size, self.output_size]],
                activation = nn.Softplus,
                output_activation = [None, Exp],
                allow_broadcast = self.allow_broadcast,
                use_cuda = self.use_cuda,
            )
        elif self.label_type == 'discrete':
            self.encoder_zy_y = MLP(
                [self.z_dim] + hidden_sizes + [self.output_size],
                activation = nn.Softplus,
                output_activation = nn.Softplus,
                allow_broadcast = self.allow_broadcast,
                use_cuda = self.use_cuda,
            )

        self.encoder_zy_z = MLP(
            [self.z_dim] + hidden_sizes + [[self.z_dim,self.z_dim]],
            activation =  nn.Softplus,
            output_activation = [None,Exp],
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )

        self.encoder_y_mask = MLP(
            [self.output_size] + hidden_sizes + [self.input_size],
            activation =  nn.Softplus,
            output_activation = None,
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )

        self.encoder_zn = MLP(
            [self.input_size] + hidden_sizes + [[z_dim, 1]],
            activation =  nn.Softplus,
            output_activation = [None, Exp],
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )

        self.decoder_zy = MLP(
            [self.output_size + self.z_dim] + hidden_sizes + [[z_dim, z_dim]],
            activation =  nn.Softplus,
            output_activation = [None,Exp],
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )

        self.decoder_concentrate = MLP(
            [self.z_dim + self.z_dim] + hidden_sizes + [self.input_size],
            activation = nn.Softplus,
            output_activation = None,
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )

        self.encoder_gate = MLP(
            [z_dim + self.z_dim] + hidden_sizes + [[self.input_size, self.input_size]],
            activation = nn.Softplus,
            output_activation = [None,Exp],
            allow_broadcast = self.allow_broadcast,
            use_cuda = self.use_cuda,
        )

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()
    
    def cutoff(self, xs, thresh = None):
        if thresh is None:
            eps = torch.finfo(xs.dtype).eps
            xs = xs.clamp(min=eps)
        else:
            xs = nn.Threshold(thresh, thresh)(xs)
        
        return xs

    def softmax(self,xs):
        soft_enc = nn.Softmax(dim=1)
        xs = soft_enc(xs)
        xs = clamp_probs(xs)
        return xs

    def sigmoid(self,xs):
        sigm_enc = nn.Sigmoid()
        xs = sigm_enc(xs)
        xs = clamp_probs(xs)
        return xs

    def softmax_logit(self,xs):
        eps = torch.finfo(xs.dtype).eps
        xs = self.softmax(xs)
        xs = torch.logit(xs, eps=eps)
        return xs
    
    def logit(self,xs):
        eps = torch.finfo(xs.dtype).eps
        xs = torch.logit(xs, eps=eps)
        return xs

    def dirimulti_param(self,xs):
        xs = self.dirimulti_mass * self.sigmoid(xs)
        return xs

    def multi_param(self,xs):
        xs = self.softmax(xs)
        return xs

    def model(self, xs, ys = None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module('scc', self)

        total_count = pyro.param("inverse_dispersion", 10.0 * xs.new_ones(self.input_size), constraint=constraints.positive)

        eps = torch.finfo(xs.dtype).eps
        batch_size = xs.size(0)
        options = dict(dtype = xs.dtype, device = xs.device)
        with pyro.plate('data'):
            ###############################################
            # p(y)
            if ys is not None:   
                alpha_prior = torch.ones(batch_size, self.output_size, **options) / (
                    1.9 * self.output_size
                )
                if self.label_type == 'categorical':
                    ys_ = pyro.sample('y', dist.OneHotCategorical(alpha_prior), obs = ys)
                elif self.label_type == 'compositional':
                    if torch.any(ys<eps):
                        ys = clamp_probs(ys)
                    ys = ft.normalize(ys, 1, 1)
                    ys_ = pyro.sample('y', dist.Dirichlet(concentration=alpha_prior), obs = ys)
                elif self.label_type == 'rate':
                    if torch.any(ys<eps):
                        ys = clamp_probs(ys)
                    prior_alpha = torch.ones(batch_size, self.output_size, **options)
                    prior_beta = torch.ones(batch_size, self.output_size, **options)
                    ys_ = pyro.sample('y', dist.Beta(prior_alpha, prior_beta).to_event(1), obs = ys)
                elif self.label_type == 'onehot':
                    prior_alpha = torch.ones(batch_size, self.output_size, **options)
                    prior_beta = torch.ones(batch_size, self.output_size, **options)
                    ys_ = pyro.sample('y', dist.BetaBinomial(prior_alpha, prior_beta).to_event(1), obs = ys)
                elif self.label_type == 'real':
                    prior_loc = torch.zeros(batch_size, self.output_size, **options)
                    prior_scale = torch.ones(batch_size, self.output_size, **options)
                    ys_ = pyro.sample('y', dist.Normal(prior_loc, prior_scale).to_event(1), obs = ys)
                elif self.label_type == 'discrete':
                    alpha_prior = torch.ones(batch_size, self.output_size, **options)
                    ys_ = pyro.sample('y', dist.Poisson(alpha_prior).to_event(1), obs = ys)
            else:
                alpha_prior = torch.ones(batch_size, self.output_size, **options) / (
                    1.9 * self.output_size
                )
                if self.label_type == 'categorical':
                    ys = pyro.sample('y', dist.OneHotCategorical(alpha_prior))
                elif self.label_type == 'compositional':
                    ys = pyro.sample('y', dist.Dirichlet(concentration=alpha_prior))
                elif self.label_type == 'rate':
                    prior_alpha = torch.ones(batch_size, self.output_size, **options)
                    prior_beta = torch.ones(batch_size, self.output_size, **options)
                    ys = pyro.sample('y', dist.Beta(prior_alpha, prior_beta).to_event(1))
                elif self.label_type == 'onehot':
                    prior_alpha = torch.ones(batch_size, self.output_size, **options)
                    prior_beta = torch.ones(batch_size, self.output_size, **options)
                    ys = pyro.sample('y', dist.BetaBinomial(prior_alpha, prior_beta).to_event(1))
                elif self.label_type == 'real':
                    prior_loc = torch.zeros(batch_size, self.output_size, **options)
                    prior_scale = torch.ones(batch_size, self.output_size, **options)
                    ys = pyro.sample('y', dist.Normal(prior_loc, prior_scale).to_event(1))
                elif self.label_type == 'discrete':
                    alpha_prior = torch.ones(batch_size, self.output_size, **options)
                    ys = pyro.sample('y', dist.Poisson(alpha_prior).to_event(1))

            # p(dy)
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            zy_zs = pyro.sample('zy_z', dist.Normal(prior_loc, prior_scale).to_event(1))

            # p(zy | y, dy)
            zy_loc,zy_scale = self.decoder_zy([ys, zy_zs])
            zys = pyro.sample('zy', dist.Normal(zy_loc, zy_scale).to_event(1))

            ###############################################
            # p(zn)
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            zns = pyro.sample('zn', dist.Normal(prior_loc, prior_scale).to_event(1))

            ###############################################
            # p(a | zys, zns)
            zs = [zys, zns]
            concentrate = self.decoder_concentrate(zs)

            if self.dist_model == 'dmm':
                concentrate = self.dirimulti_param(concentrate)
                theta = dist.DirichletMultinomial(total_count=1, concentration=concentrate).mean
            elif self.dist_model == 'mm':
                probs = self.multi_param(concentrate)
                theta = dist.Multinomial(total_count=1, probs=probs).mean

            # zero-inflation model
            if self.use_zeroinflate:
                gate_loc = self.gate_prior * torch.ones(batch_size, self.input_size, **options)
                gate_scale = torch.ones(batch_size, self.input_size, **options)
                gate_logits = pyro.sample('gate_logit', dist.Normal(gate_loc, gate_scale).to_event(1))
                gate_probs = self.sigmoid(gate_logits)

                theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                theta = logits_to_probs(theta)

                if self.delta > 0:
                    ones = torch.zeros(batch_size, self.input_size, **options)
                    ones[xs>0] = 1
                    with pyro.poutine.scale(scale = self.delta):
                        ones = pyro.sample('one', dist.Binomial(probs=1-gate_probs).to_event(1), obs=ones)

            if self.loss_func=='negbinomial':
                pyro.sample('x', dist.NegativeBinomial(total_count=total_count, probs=theta).to_event(1), obs=xs)
            elif self.loss_func=='poisson':
                rate = xs.sum(1).unsqueeze(-1) * theta
                pyro.sample('x', dist.Poisson(rate=rate).to_event(1), obs=xs)
            else:
                pyro.sample('x', dist.Multinomial(total_count=int(1e8), probs=theta), obs=xs)

    def guide(self, xs, ys = None):
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data'):
            ####################################
            # q(zy | x)
            if self.use_mask and (not ys is None):
                my_theta = self.encoder_y_mask(ys)
                my_theta = self.sigmoid(my_theta)
                mys = pyro.sample('m_y', dist.Binomial(probs=my_theta).to_event(1))
                zy_loc, zy_scale = self.encoder_zy(xs * mys)
            else:
                zy_loc, zy_scale = self.encoder_zy(xs)
            zys = pyro.sample('zy', dist.Normal(zy_loc, zy_scale).to_event(1))

            # q(zn | x)
            zn_loc, zn_scale = self.encoder_zn(xs)
            zn_scale = self.cutoff(zn_scale)
            zns = pyro.sample('zn', dist.Normal(zn_loc, zn_scale).to_event(1))

            ####################################
            # q(y | zy)
            alpha_y = self.encoder_zy_y(zys)
            if ys is None:
                if self.label_type == 'categorical':
                    ys = pyro.sample('y', dist.OneHotCategorical(logits=alpha_y))
                elif self.label_type == 'compositional':
                    alpha_y = self.cutoff(alpha_y)
                    ys = pyro.sample('y', dist.Dirichlet(concentration=alpha_y))
                elif self.label_type == 'rate':
                    alpha_y[0] = self.cutoff(alpha_y[0])
                    alpha_y[1] = self.cutoff(alpha_y[1])
                    ys = pyro.sample('y', dist.Beta(alpha_y[0], alpha_y[1]).to_event(1))
                elif self.label_type == 'onehot':
                    alpha_y[0] = self.cutoff(alpha_y[0])
                    alpha_y[1] = self.cutoff(alpha_y[1])
                    ys = pyro.sample('y', dist.BetaBinomial(alpha_y[0], alpha_y[1]).to_event(1))
                elif self.label_type == 'real':
                    alpha_y[1] = self.cutoff(alpha_y[1])
                    ys = pyro.sample('y', dist.Normal(alpha_y[0], alpha_y[1]).to_event(1))
                elif self.label_type == 'discrete':
                    alpha_y = self.cutoff(alpha_y)
                    ys = pyro.sample('y', dist.Poisson(alpha_y).to_event(1))
            
            # q(z | zy)
            zy_z_loc,zy_z_scale = self.encoder_zy_z(zys)
            zy_zs = pyro.sample('zy_z', dist.Normal(zy_z_loc, zy_z_scale).to_event(1))

            ####################################
            # q(gate | xs)
            if self.use_zeroinflate:
                loc,scale = self.encoder_gate([zys,zns])
                gates_logit = pyro.sample('gate_logit', dist.Normal(loc,scale).to_event(1))

    def classifier_state(self, xs):
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the cell(s)
        zys,_ = self.encoder_zy(xs)
        alpha = self.encoder_zy_y(zys)

        if self.label_type in ['rate','onehot']:
            alpha[0] = self.cutoff(alpha[0])
            alpha[1] = self.cutoff(alpha[1])
            alpha = alpha[0]/(alpha[0]+alpha[1])
        elif self.label_type == 'real':
            alpha = alpha[0]

        # get the index (digit) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the digit(s) to one-hot tensor(s)
        ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)

        return ys
    
    def classifier_state_score(self, xs):
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the cell(s)
        zys,_ = self.encoder_zy(xs)
        alpha = self.encoder_zy_y(zys)
        
        if self.label_type in ['rate','onehot']:
            alpha[0] = self.cutoff(alpha[0])
            alpha[1] = self.cutoff(alpha[1])
            alpha = alpha[0]/(alpha[0]+alpha[1])
        elif self.label_type == 'real':
            alpha = alpha[0]

        if self.label_type in ['categorical','compositional']:
            alpha = self.softmax(alpha)

        return alpha

    def state_label(self, ys):
        _, ind = torch.topk(ys, 1)
        ys = self.label_names[ind]
        return ys

    def latent_embedding(self, xs, ys=None, ks=None, use_sampling = False):
        zys,_ = self.encoder_zy(xs)

        return zys

    def generate_expression(self, xs, mute_label=False, mute_noise=False, library_size=1, use_sampling=False, use_gate=False):
        zys,_ = self.encoder_zy(xs)
        if mute_label:
            zys = torch.zeros_like(zys)

        zns,_ = self.encoder_zn(xs)
        if mute_noise:
            zns = torch.zeros_like(zns)

        zs = [zys,zns]
        concentrate = self.decoder_concentrate(zs)

        if use_gate:
            gate_logits,_ = self.encoder_gate(zs)
            gates = self.sigmoid(gate_logits)

        if self.dist_model=='dmm':
            concentrate = self.dirimulti_param(concentrate)
            if use_sampling:
                xs = pyro.sample('x',dist.DirichletMultinomial(concentration=concentrate, total_count=int(library_size), is_sparse=True))
            else:
                xs = dist.DirichletMultinomial(concentration=concentrate, total_count=1).mean
                if use_gate:
                    xs = (1-gates) * xs                  
        elif self.dist_model=='mm':
            probs = self.multi_param(concentrate)
            if use_sampling:
                xs = pyro.sample('x',dist.Multinomial(probs=probs, total_count=int(library_size)))
            else:
                xs = dist.Multinomial(probs=probs, total_count=1).mean
                if use_gate:
                    xs = (1-gates) * xs                   

        xs = xs * library_size

        return xs

    def mute_expression(self, xs, mute_label_names=list(), mute_noise=False, library_size=1, use_sampling=False, use_gate=False):
        if len(mute_label_names)>0:
            i = [j for j in np.arange(len(self.label_names)) if self.label_names[j] in mute_label_names]
            ys = self.classifier_state_score(xs)
            ys[:,i]=0
            zy_zs = torch.zeros(xs.shape[0], self.z_dim)
            zys,_ = self.decoder_zy([ys,zy_zs])
        else:
            zys,_ = self.encoder_zy(xs)

        zns,_ = self.encoder_zn(xs)
        if mute_noise:
            zns = torch.zeros_like(zns)

        zs = [zys,zns]
        concentrate = self.decoder_concentrate(zs)

        if use_gate:
            gate_logits,_ = self.encoder_gate(zs)
            gates = self.sigmoid(gate_logits)

        if self.dist_model=='dmm':
            concentrate = self.dirimulti_param(concentrate)
            if use_sampling:
                xs = pyro.sample('x',dist.DirichletMultinomial(concentration=concentrate, total_count=int(library_size), is_sparse=True))
            else:
                xs = dist.DirichletMultinomial(concentration=concentrate, total_count=1).mean
                if use_gate:
                    xs = (1-gates) * xs                  
        elif self.dist_model=='mm':
            probs = self.multi_param(concentrate)
            if use_sampling:
                xs = pyro.sample('x',dist.Multinomial(probs=probs, total_count=int(library_size)))
            else:
                xs = dist.Multinomial(probs=probs, total_count=1).mean
                if use_gate:
                    xs = (1-gates) * xs                   

        xs = xs * library_size

        return xs

    def spike_expression(self, xs, spike_label_names=list(), mute_noise=False, library_size=1, use_sampling=False, use_gate=False):
        mute_label_names = [l for l in self.label_names if l not in spike_label_names]
        
        if len(mute_label_names)>0:
            i = [j for j in np.arange(len(self.label_names)) if self.label_names[j] in mute_label_names]
            ys = self.classifier_state_score(xs)
            ys[:,i]=0
            zy_zs = torch.zeros(xs.shape[0], self.z_dim)
            zys,_ = self.decoder_zy([ys,zy_zs])
        else:
            zys,_ = self.encoder_zy(xs)

        zns,_ = self.encoder_zn(xs)
        if mute_noise:
            zns = torch.zeros_like(zns)

        zs = [zys,zns]
        concentrate = self.decoder_concentrate(zs)

        if use_gate:
            gate_logits,_ = self.encoder_gate(zs)
            gates = self.sigmoid(gate_logits)

        if self.dist_model=='dmm':
            concentrate = self.dirimulti_param(concentrate)
            if use_sampling:
                xs = pyro.sample('x',dist.DirichletMultinomial(concentration=concentrate, total_count=int(library_size), is_sparse=True))
            else:
                xs = dist.DirichletMultinomial(concentration=concentrate, total_count=1).mean
                if use_gate:
                    xs = (1-gates) * xs                  
        elif self.dist_model=='mm':
            probs = self.multi_param(concentrate)
            if use_sampling:
                xs = pyro.sample('x',dist.Multinomial(probs=probs, total_count=int(library_size)))
            else:
                xs = dist.Multinomial(probs=probs, total_count=1).mean
                if use_gate:
                    xs = (1-gates) * xs                   

        xs = xs * library_size

        return xs

    def model_classify(self, xs, ys = None):
        """
        this model is used to add auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module('scc', self)
        eps = torch.finfo(xs.dtype).eps
        # inform pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data'):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                zys_loc,_ = self.encoder_zy(xs)
                alpha_y = self.encoder_zy_y(zys_loc)
                with pyro.poutine.scale(scale = 1 * self.aux_loss_multiplier):
                    if self.label_type == 'categorical':
                        ys_aux = pyro.sample('y_aux', dist.OneHotCategorical(logits=alpha_y), obs = ys)
                    elif self.label_type == 'compositional':
                        if torch.any(ys<eps):
                            ys = clamp_probs(ys)
                        ys = ft.normalize(ys,1,1)
                        alpha_y = self.cutoff(alpha_y)
                        ys_aux = pyro.sample('y_aux', dist.Dirichlet(concentration=alpha_y), obs = ys)
                    elif self.label_type == 'rate':
                        if torch.any(ys<eps):
                            ys = clamp_probs(ys)
                        alpha_y[0] = self.cutoff(alpha_y[0])
                        alpha_y[1] = self.cutoff(alpha_y[1])
                        ys_aux = pyro.sample('y_aux', dist.Beta(alpha_y[0], alpha_y[1]).to_event(1), obs = ys)
                    elif self.label_type == 'onehot':
                        alpha_y[0] = self.cutoff(alpha_y[0])
                        alpha_y[1] = self.cutoff(alpha_y[1])
                        ys_aux = pyro.sample('y_aux', dist.BetaBinomial(alpha_y[0], alpha_y[1]).to_event(1), obs = ys)
                    elif self.label_type == 'real':
                        alpha_y[1] = self.cutoff(alpha_y[1])
                        ys_aux = pyro.sample('y_aux', dist.Normal(alpha_y[0], alpha_y[1]).to_event(1), obs = ys)
                    elif self.label_type == 'discrete':
                        alpha_y = self.cutoff(alpha_y)
                        ys_aux = pyro.sample('y_aux', dist.Poisson(alpha_y).to_event(1), obs = ys)

    def guide_classify(self, xs, ys = None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass


def run_inference_for_epoch(sup_data_loader, unsup_data_loader, losses, use_cuda=True):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)

    # compute number of batches for an epoch
    sup_batches = len(sup_data_loader)
    unsup_batches = len(unsup_data_loader) if unsup_data_loader is not None else 0

    # initialize variables to store loss values
    epoch_losses_sup = [0.0] * num_losses
    epoch_losses_unsup = [0.0] * num_losses

    # setup the iterators for training data loaders
    sup_iter = iter(sup_data_loader)
    unsup_iter = iter(unsup_data_loader) if unsup_data_loader is not None else None

    # supervised data
    for i in range(sup_batches):
        # extract the corresponding batch
        (xs, ys) = next(sup_iter)

        # run the inference for each loss with supervised data as arguments
        for loss_id in range(num_losses):
            new_loss = losses[loss_id].step(xs, ys)
            epoch_losses_sup[loss_id] += new_loss

    # unsupervised data
    if unsup_data_loader is not None:
        for i in range(unsup_batches):
            # extract the corresponding batch
            (xs, ys) = next(unsup_iter)

            # run the inference for each loss with unsupervised data as arguments
            for loss_id in range(num_losses):
                new_loss = losses[loss_id].step(xs)
                epoch_losses_unsup[loss_id] += new_loss

    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup


def get_accuracy(data_loader, classifier_fn):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions, actuals = [], []

    # use the appropriate data loader
    for (xs, ys) in data_loader:
        # use classification function to compute all predictions for each batch
        predictions.append(classifier_fn(xs))
        actuals.append(ys)

    # compute the number of accurate predictions
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    _, y = torch.topk(actuals, 1)
    _, yhat = torch.topk(predictions, 1)
    y = y.detach().cpu().numpy()
    yhat = yhat.detach().cpu().numpy()
    accuracy = accuracy_score(y, yhat)
    f1_macro = f1_score(y, yhat, average='macro')
    f1_weighted = f1_score(y, yhat, average='weighted')
    precision = precision_score(y, yhat, average='macro')
    recall = recall_score(y, yhat, average='macro')
    mcc = matthews_corrcoef(y, yhat)

    return accuracy, f1_macro, f1_weighted, precision, recall, mcc

def main(args):
    """
    run inference for scClassifier

    :param args: arguments for scClassifier
    :return: None
    """
    if args.seed is not None:
        pyro.set_rng_seed(args.seed)

    if args.float64:
        torch.set_default_dtype(torch.float64)

    # prepare dataloaders
    data_loaders = {'sup':None, 'unsup':None, 'valid':None}
    sup_num, unsup_num = 0, 0
    if args.sup_data_file is not None:
        data_loaders['sup'], data_loaders['valid'] = setup_data_loader(
            SingleCellCached, args.sup_data_file, args.sup_label_file,  
            'sup', args.validation_fold, args.log_transform, args.expm1, args.cuda, args.float64, args.batch_size
        )
        sup_num = len(data_loaders['sup'])
    if args.unsup_data_file is not None:
        data_loaders['unsup'], _ = setup_data_loader(
            SingleCellCached, args.unsup_data_file, args.unsup_label_file,  
            'unsup', 0, args.log_transform, args.expm1, args.cuda, args.float64, args.batch_size
        )
        unsup_num = len(data_loaders['unsup'])

    if args.validation_fold==0:
        output_size = data_loaders['sup'].dataset.num_classes
        input_size = data_loaders['sup'].dataset.data.shape[1]
        label_names = data_loaders['sup'].dataset.label_names
    else:
        output_size = data_loaders['sup'].dataset.dataset.num_classes
        input_size = data_loaders['sup'].dataset.dataset.data.shape[1]
        label_names = data_loaders['sup'].dataset.dataset.label_names

    use_mask = False
    if args.mask:
        use_mask = True

    dist_model = 'mm'
    if args.use_dirichlet:
        dist_model = 'dmm'

    # batch_size: number of cells (and labels) to be considered in a batch
    scc = scClassifier(
        output_size=output_size,
        input_size=input_size,
        z_dim=args.z_dim,
        hidden_layers=args.hidden_layers,
        label_names=label_names,
        use_cuda=args.cuda,
        config_enum=args.enum_discrete,
        aux_loss_multiplier=args.aux_loss_multiplier,
        use_mask = use_mask,
        mask_alpha=args.mask_alpha,
        mask_beta=args.mask_beta,
        dist_model= dist_model,
        use_zeroinflate=args.zero_inflation,
        gate_prior=args.gate_prior,
        delta=args.delta,
        loss_func=args.likelihood,
        dirimulti_mass=args.dirichlet_mass,
        label_type=args.label_type
    )

    # setup the optimizer
    adam_params = {'lr': args.learning_rate, 'betas':(args.beta_1, 0.999), 'weight_decay': 0.005}
    #optimizer = Adam(adam_params)
    optimizer = torch.optim.Adam
    decayRate = args.decay_rate
    scheduler = ExponentialLR({'optimizer': optimizer, 'optim_args': adam_params, 'gamma': decayRate})

    pyro.clear_param_store()

    # set up the loss(es) for inference, wrapping the guide in config_enumerate builds the loss as a sum
    # by enumerating each class label form the sampled discrete categorical distribution in the model
    guide = config_enumerate(scc.guide, args.enum_discrete, expand = True)
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=1, strict_enumeration_warning=False)
    loss_basic = SVI(scc.model, guide, scheduler, loss = elbo)

    # build a list of all losses considered
    losses = [loss_basic]

    # aux_loss: whether to use the auxiliary loss from NIPS 14 papers (Kingma et al)
    if args.aux_loss:
        elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
        loss_aux = SVI(scc.model_classify, scc.guide_classify, scheduler, loss = elbo)
        losses.append(loss_aux)

    try:
        # setup the logger if a filename is provided
        logger = open(args.logfile, 'w') if args.logfile else None

        # initializing local variables to maintain the best validation accuracy
        # seen across epochs over the supervised training set
        # and the corresponding testing set and the state of the networks
        best_loss = 1e100
        best_valid_acc = 0.0
        best_valid_f1m = 0.0

        asso_valid_f1m = 0.0
        asso_valid_f1w = 0.0
        asso_valid_pre = 0.0
        asso_valid_rec = 0.0
        asso_valid_mcc = 0.0
        
        asso_unsup_acc = 0.0
        asso_unsup_f1m = 0.0
        asso_unsup_f1w = 0.0
        asso_unsup_pre = 0.0
        asso_unsup_rec = 0.0
        asso_unsup_mcc = 0.0
        
        tr_start=tm.time()
        # run inference for a certain number of epochs
        for i in range(0, args.num_epochs):
            ep_tr_start = tm.time()

            # get the losses for an epoch
            epoch_losses_sup, epoch_losses_unsup = run_inference_for_epoch(
                data_loaders['sup'], data_loaders['unsup'], losses, args.cuda
            )

            scc.inverse_dispersion = pyro.param('inverse_dispersion').data

            # compute average epoch losses i.e. losses per example
            avg_epoch_losses_sup = map(lambda v: v / sup_num, epoch_losses_sup)
            avg_epoch_losses_unsup = map(lambda v: v / unsup_num, epoch_losses_unsup) if unsup_num > 0 else [0] * len(epoch_losses_unsup)
            avg_epoch_losses_sup = map(lambda v: "{:.4f}".format(v), avg_epoch_losses_sup)
            avg_epoch_losses_unsup = map(lambda v: "{:.4f}".format(v), avg_epoch_losses_unsup)

            # store the loss
            str_loss_sup = " ".join(map(str, avg_epoch_losses_sup))
            str_loss_unsup = " ".join(map(str, avg_epoch_losses_unsup))

            str_print = "{} epoch: avg losses {}".format(
                i+1, "{} {}".format(str_loss_sup, str_loss_unsup)
            )

            if args.label_type == 'categorical' and args.validation_fold>0:
                validation_accuracy, validation_f1_macro, validation_f1_weighted, validation_precision, validation_recall, validation_mcc = get_accuracy(
                    data_loaders["valid"], scc.classifier_state
                )

                str_print += " validation accuracy {:.4f}".format(validation_accuracy)
                str_print += " F1 {:.4f}(macro) {:.4f}(weighted)".format(validation_f1_macro, validation_f1_weighted)
                str_print += " precision {:.4f} recall {:.4f}".format(validation_precision, validation_recall)
                str_print += " mcc {:.4f}".format(validation_mcc)

                if (args.unsup_label_file is not None) and (args.unsup_data_file is not None):
                    unsup_accuracy, unsup_f1_macro, unsup_f1_weighted, unsup_precision, unsup_recall, unsup_mcc = get_accuracy(
                        data_loaders['unsup'], scc.classifier_state
                    )            

            ep_tr_time = tm.time() - ep_tr_start
            str_print += " elapsed {:.4f} seconds".format(ep_tr_time)

            if args.label_type == 'categorical' and args.validation_fold>0:
                # update the best validation accuracy and the state of the parent 
                # module (including the networks)
                if best_valid_acc <= validation_accuracy:
                    do_update = False
                    if best_valid_acc < validation_accuracy:
                        do_update = True
                    elif best_valid_f1m < validation_f1_macro:
                        do_update = True

                    if do_update:
                        best_valid_acc = validation_accuracy
                        best_valid_f1m = validation_f1_macro
                    
                        asso_valid_f1m = validation_f1_macro
                        asso_valid_f1w = validation_f1_weighted
                        asso_valid_pre = validation_precision
                        asso_valid_rec = validation_recall
                        asso_valid_mcc = validation_mcc

                        if (args.unsup_label_file is not None) and (args.unsup_data_file is not None):
                            asso_unsup_acc = unsup_accuracy
                            asso_unsup_f1m = unsup_f1_macro
                            asso_unsup_f1w = unsup_f1_weighted
                            asso_unsup_pre = unsup_precision
                            asso_unsup_rec = unsup_recall
                            asso_unsup_mcc = unsup_mcc

                        if args.save_model is not None:
                            if args.best_accuracy:
                                torch.save(scc, args.save_model)

            if i%args.decay_epochs == 0:
                scheduler.step() 

            if (i+1) == args.num_epochs:
                if args.save_model is not None:
                    if not args.best_accuracy:
                        if best_loss * sup_num > epoch_losses_sup[0]:
                            best_loss = epoch_losses_sup[0] / sup_num
                            torch.save(scc, args.save_model)

            print_and_log(logger, str_print)

        tr_time=tm.time()-tr_start
        if args.runtime:
            print('running time: {} secs'.format(tr_time))

        print_and_log(
            logger,
            "best validation accuracy {:.4f}".format(
                best_valid_acc
            ),
        )
        if (args.unsup_label_file is not None) and (args.unsup_data_file is not None):
            print_and_log(
                logger,
                "unsup accuracy: {:.4f} \nF1: {:.4f}(macro) {:.4f}(weighted) \nprecision {:.4f} recall {:.4f} \nmcc {:.4f}".format(
                    asso_unsup_acc, asso_unsup_f1m, asso_unsup_f1w, asso_unsup_pre, asso_unsup_rec, asso_unsup_mcc
                ),
            )
        else:
            print_and_log(
                logger,
                "F1: {:.4f}(macro) {:.4f}(weighted) \nprecision {:.4f} recall {:.4f} \nmcc {:.4f}".format(
                    asso_valid_f1m, asso_valid_f1w, asso_valid_pre, asso_valid_rec, asso_valid_mcc
                ),
            )
    finally:
        # close the logger file object if we opened it earlier
        if args.logfile:
            logger.close()

EXAMPLE_RUN = (
    "example run: python scClassifier.py --seed 0 --cuda -n 50 --aux-loss -alm 46 -enum parallel "
    "--sup-data-file <sup_data_file> --sup-label-file <sup_label_file> "
    "--unsup-data-file [unsup_data_file] --unsup-label-file [sup_label_file] "
    "-zd 50 -hl 500 -lr 0.0001 -b1 0.99 -bs 100 -log ./tmp.log"
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="scClassifier\n{}".format(EXAMPLE_RUN))

    parser.add_argument(
        "--cuda", action="store_true", help="use GPU(s) to speed up training"
    )
    parser.add_argument(
        "--jit", action="store_true", help="use PyTorch jit to speed up training"
    )
    parser.add_argument(
        "-n", "--num-epochs", default=40, type=int, help="number of epochs to run"
    )
    parser.add_argument(
        "--aux-loss",
        action="store_true",
        help="whether to use the auxiliary loss from NIPS 14 paper "
        "(Kingma et al). It is not used by default ",
    )
    parser.add_argument(
        "-alm",
        "--aux-loss-multiplier",
        default=46,
        type=float,
        help="the multiplier to use with the auxiliary loss",
    )
    parser.add_argument(
        "-enum",
        "--enum-discrete",
        default="parallel",
        help="parallel, sequential or none. uses parallel enumeration by default",
    )
    parser.add_argument(
        "--sup-data-file",
        default=None,
        type=str,
        help="the data file of the supervised data",
    )
    parser.add_argument(
        "--sup-label-file",
        default=None,
        type=str,
        help="the label file of the supervised data",
    )
    parser.add_argument(
        "--unsup-data-file",
        default=None,
        type=str,
        help="the data file of the unsupervised data",
    )
    parser.add_argument(
        "--unsup-label-file",
        default=None,
        type=str,
        help="the label file of the unsupervised data",
    )
    parser.add_argument(
        "--label-type",
        default='categorical',
        type=str,
        choices=['categorical','compositional','onehot','rate','real','discrete'],
        help="specify the type of label variable",
    )
    parser.add_argument(
        "-delta",
        "--delta",
        default=0,
        type=float,
        help="penalty weight for zero inflation loss",
    )
    parser.add_argument(
        "-64",
        "--float64",
        action="store_true",
        help="use double float precision",
    )
    parser.add_argument(
        "-lt",
        "--log-transform",
        action="store_true",
        help="run log-transform on count data",
    )
    parser.add_argument(
        "-cv",
        "--validation-fold",
        default=10,
        type=float,
        help="one of the folds of the supervised data for validation",
    )
    parser.add_argument(
        "-zd",
        "--z-dim",
        default=100,
        type=int,
        help="size of the tensor representing the latent variable z "
        "variable (handwriting style for our MNIST dataset)",
    )
    parser.add_argument(
        "-hl",
        "--hidden-layers",
        nargs="+",
        default=[300],
        type=int,
        help="a tuple (or list) of MLP layers to be used in the neural networks "
        "representing the parameters of the distributions in our model",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.0001,
        type=float,
        help="learning rate for Adam optimizer",
    )
    parser.add_argument(
        "-dr",
        "--decay-rate",
        default=0.9,
        type=float,
        help="decay rate for Adam optimizer",
    )
    parser.add_argument(
        "-de",
        "--decay-epochs",
        default=20,
        type=int,
        help="decay learning rate every #epochs",
    )
    parser.add_argument(
        "-b1",
        "--beta-1",
        default=0.95,
        type=float,
        help="beta-1 parameter for Adam optimizer",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        default=1000,
        type=int,
        help="number of images (and labels) to be considered in a batch",
    )
    parser.add_argument(
        "-expm1",
        "--expm1",
        action="store_true",
        help="turn on exponential transformation",
    )
    parser.add_argument(
        "-mo",
        "--mask",
        action="store_true",
        help="turn on masking",
    )
    parser.add_argument(
        "-mal",
        "--mask-alpha",
        default=1,
        type=float,
        help="Beta prior distribution parameter alpha for mask",
    )
    parser.add_argument(
        "-mbe",
        "--mask-beta",
        default=10,
        type=float,
        help="Beta prior distribution parameter beta for mask",
    )
    parser.add_argument(
        "-gp",
        "--gate-prior",
        default=0.7,
        type=float,
        help="gate prior for zero-inflated model",
    )
    parser.add_argument(
        "-likeli",
        "--likelihood",
        default='negbinomial',
        type=str,
        choices=['negbinomial','multinomial','poisson'],
        help="specify the distribution likelihood function",
    )
    parser.add_argument(
        "-dirichlet",
        "--use-dirichlet",
        action="store_true",
        help="use Dirichlet distribution over gene compositional",
    )
    parser.add_argument(
        "-mass",
        "--dirichlet-mass",
        default=0.01,
        type=float,
        help="mass param for dirichlet model",
    )
    parser.add_argument(
        "-zi",
        "--zero-inflation",
        action="store_true",
        help="use zero-inflated estimation",
    )
    parser.add_argument(
        "-rt",
        "--runtime",
        action="store_true",
        help="print running time",
    )
    parser.add_argument(
        "-log",
        "--logfile",
        default="./tmp.log",
        type=str,
        help="filename for logging the outputs",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for controlling randomness in this example",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        type=str,
        help="path to save model for prediction",
    )
    parser.add_argument(
        "-ba",
        "--best-accuracy",
        action="store_true",
        help="save the model with best classification accuracy",
    )
    args = parser.parse_args()

    assert (
        (args.sup_data_file is not None) and (os.path.exists(args.sup_data_file))
    ), "sup_data_file must be provided"
    assert (
        (args.sup_label_file is not None) and (os.path.exists(args.sup_label_file))
    ), "sup_data_file must be provided"
    assert (
        (args.validation_fold >= 0)
    ), "fold of the supervised data used for validation should be greater than 0"

    if args.validation_fold > 0:
        args.validation_fold = 1. / args.validation_fold
    
    main(args)
