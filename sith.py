# newest version of TILT
# uses pytorch operations

import torch
from torch import nn as nn

import numpy as np
from math import factorial

#####################
# functions for TILT
#####################
def _calc_tau_star(tau_0, k, c, ntau):
    # tau_0=.1, k=4, c=.1, ntau=100

    # [PER CHECK]
    #ntau = ntau + 2*k
    # I commented out the above line to make ntau a more accurate parameter. 
    # I do not thing this actually has any lasting changes anywher elsein the
    # code, just let me know. 
    tau_star = tau_0*(1+c)**torch.arange(-k, ntau+k).type(torch.DoubleTensor)
    s = k/tau_star 
    return tau_star, s


def _calc_D(s):
    # calc all the differences
    s0__1 = s[1:-1] - s[:-2]
    s1_0 = s[2:] - s[1:-1]
    s1__1 = s[2:] - s[:-2]

    # calc the -1, 0, and 1 diagonals
    A = -((s1_0/s1__1)/s0__1)
    B = -((s0__1/s1__1)/s1_0) + (s1_0/s1__1)/s0__1
    C = (s0__1/s1__1)/s1_0
    
    # create the matrix
    D = torch.zeros(s.shape[0],s.shape[0]).type(torch.DoubleTensor)
    s2r = torch.arange(D.shape[0])
    s2c = torch.arange(D.shape[0])
    
    D[(s2r + 1)[:-2], s2c[:-2]] = A
    
    D[(s2r + 1)[:-2], (s2c+1)[:-2]] = B
    D[(s2r + 1)[:-2], (s2c+2)[:-2]] = C
    D = D.transpose(1,0)
    return D

def _calc_invL(s, k):
    # easier to do with matrices than ndarray to keep dimensions straight
    D = _calc_D(s)
    diag = torch.diag(s ** (k+1))
    invL = ((((-1.) ** k) /
            factorial(k)) * 
                        torch.mm(torch.matrix_power(D,k),
                                  torch.diag(s ** (k+1.))
                                    ))[:, k:-k]
    # return as ndarray
    return invL.transpose(1,0)


class SITH(nn.Module):
    """SITH implementation."""
    def __init__(self, in_features, tau_0=.1, dt=None, k=4, c=.1,
                 alpha=1.0, g=0, ntau=100, s_toskip=0, T_every=8, 
                 ttype=torch.DoubleTensor):
        """The SITH layer has a lot of different parameters that allow you to fine 
        tune exactly how compressed you want the historical representation to be.

        Parameters
        ----------
            in_features: int
                Number of tracked features
            tau_0: float (default = 1)
                The center of the first receptive field in inverse-Lapace space. The
                presentation time of each stimulus.
            dt: float (default=.1)
                The input will be presented to the representation at tau_0 / dt seconds.
                This will smooth the exponential decay, and provide a better estimate of 
                the past in the final inverse-laplace representation.
            k: int (default = 4)
                The spcificity of the receptive fields
            c: float
                The degree of historical compression. Smaller numbers means greater
                numbers of tau*s will be dedicated to tracking the more recent past
            alpha: float
            g: float (default = 0)
                A conditionditioning parameter. This will determine if the end result 
                of this layer, big T, is multiplied by tau_stars or not.  If g is 0, 
                then big T will have smaller and smaller activations further into the 
                past. If g = 1, then all taustars in big T will activate to the same 
                level at their peak. g can also be bigger than 1, but should never be 
                less than 0. 
            ntau: int (default = 100)
                The desired number of taustars in the final representation, before
                indexing with T_every
            T_every: int
                How many tau*s we skip when indexing into the inverse-laplace space
                representation, T.
        """

        super(SITH, self).__init__()
        assert(in_features > 0)
        assert(tau_0 > 0)
        assert(ntau > 0)
        assert(tau_0 >= dt)
        assert(c > 0)
        assert(k > 0)
        
        self._in_features = in_features
        self._tau_0 = tau_0
        
        self._dt = dt
        self._k = k
        self._c = c
        self._ntau = ntau
        self._T_full_ind = slice(None, None, T_every)
        self._alpha = alpha
        self._g = g
        self._torch_type = ttype

        # calc tau_star and s
        tau_star, s = _calc_tau_star(tau_0=tau_0, k=k, c=c, ntau=ntau)
        self._tau_star = tau_star[s_toskip:].type(ttype)
        self._subset_tau_star = (self._tau_star[slice(k, -k, T_every)].unsqueeze(1) ** g).repeat(1,in_features)
        
        self._output_size = self._subset_tau_star.shape[0]
        
        self._s = s[s_toskip:].type(ttype)

        # make exp diag for exponential decay
        self._sm = torch.diag(-self._s).type(ttype)

        
        # get the inverse Laplacian
        self._invL = _calc_invL(self._s, self._k).type(ttype)

        # allocate t, the internal state
        self._t = torch.zeros(self._invL.shape[1], self._in_features).type(ttype)
        self._t_changed = True
        self._id_m = torch.ones(self._s.shape[0]).type(ttype)
        
        # For each forward pass
        self._e_alph_dt = torch.exp(self._s*self._alpha*(-1*self._dt))
        self._it = torch.ones((self._s.shape[0], in_features)).type(ttype)
        self._decay = torch.unsqueeze((self._alpha*self._s)**-1 * 
                                      (self._id_m - self._e_alph_dt), 1).repeat(1,in_features)


    def cuda(self, device_id=None):
        self._invL = self._invL.cuda(device=device_id)
        self._t = self._t.cuda(device=device_id)
        self._sm = self._sm.cuda(device=device_id)
        self._it = self._it.cuda(device=device_id)
        self._s = self._s.cuda(device=device_id)
        self._id_m = self._id_m.cuda(device=device_id)
        self._e_alph_dt = self._e_alph_dt.cuda(device=device_id)
        self._decay = self._decay.cuda(device=device_id)
        self._subset_tau_star = self._subset_tau_star.cuda(device=device_id)
        self._use_cuda = True
        self._device = device_id
    
    @property
    def t(self):
        return self._t

    @property
    def k(self):
        return self._k

    @property
    def tau_star(self):
        return self._tau_star

    def forward(self, inp, dur=None):
        """
        Takes in a t state and updates with item
        This allows for calculation of little t without storing into
            `self._t`
        Returns new little t state
        """        
        
        if dur is None:
            dur = self._tau_0
        else:
            dur = dur
            
        t = self._t
        # I don't know if I should cat the output together a bunch (bad idea) 
        # or if I should construct the output as a bunch of concatentations together. 
        # For now, we will construct an output tensor, and fill it with each iteration of the loop
        output_tensor = torch.zeros(inp.shape[0], self._output_size, self._in_features).type(self._torch_type)
        if self._use_cuda:
            output_tensor = output_tensor.cuda(device=self._device)
        
        c = 0
        for item in inp.split(1, dim=0):
            # At this point, item should be of size (in_features)
            tIN = self._it * item.squeeze(0)


            ## can remove `/dt` once input isn't multiplied by dt anymore
            tIN = tIN*self._alpha
            self._t = (torch.diag(self._e_alph_dt**(dur / self._dt)).mm(self._t) +
                       self._decay * tIN * (dur / self._dt))

            # update T from t and index into it, multiply by either taustars or 1 for scaling.
            output_tensor[c, :, :] = (self._invL.mm(self._t))[self._T_full_ind, :] * self._subset_tau_star
            c += 1
        return output_tensor
    
    def reset(self):
        # reset all to zeros
        self._t.zero_()
