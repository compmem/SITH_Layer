# SITH pytorch Layer
# PyTorch version 0.1.0
# Authors: Brandon G. Jacques and Per B. Sederberg

import torch
from torch import nn as nn
from .laplace import Laplace
from math import factorial


def _calc_D(s, ttype=None):
    # calc all the differences
    s0__1 = s[1:-1] - s[:-2]
    s1_0 = s[2:] - s[1:-1]
    s1__1 = s[2:] - s[:-2]

    # calc the -1, 0, and 1 diagonals
    A = -((s1_0/s1__1)/s0__1)
    B = -((s0__1/s1__1)/s1_0) + (s1_0/s1__1)/s0__1
    C = (s0__1/s1__1)/s1_0
    
    # create the matrix
    if ttype is None:
        D = torch.zeros(s.shape[0],s.shape[0]).type(torch.DoubleTensor)
    else:
        D = torch.zeros(s.shape[0],s.shape[0]).type(ttype)
        
    s2r = torch.arange(D.shape[0])
    s2c = torch.arange(D.shape[0])
    
    D[(s2r + 1)[:-2], s2c[:-2]] = A
    
    D[(s2r + 1)[:-2], (s2c+1)[:-2]] = B
    D[(s2r + 1)[:-2], (s2c+2)[:-2]] = C
    D = D.transpose(1,0)
    return D

def _calc_Linvk(s, k, ttype=None):
    # easier to do with matrices than ndarray to keep dimensions straight
    D = _calc_D(s, ttype)
    diag = torch.diag(s ** (k+1))
    Linvk = ((((-1.) ** k) /
             factorial(k)) * 
                         torch.mm(torch.matrix_power(D,k),
                                   torch.diag(s ** (k+1.))
                                     ))[:, k:-k]
    # return as ndarray
    return Linvk.transpose(1,0)


class SITH(nn.Module):
    """SITH implementation."""
    def __init__(self, in_features, tau_min=1.0, tau_max=50, k=4,
                 alpha=1.0, g=1.0, ntau=30, T_every=1,
                 ttype=torch.FloatTensor):
        """The SITH layer has a lot of different parameters that allow you to fine 
        tune exactly how compressed you want the historical representation to be.

        Parameters
        ----------
            in_features: int
                Number of tracked features
            tau_min: float (default = 1)
                The center of the FIRST receptive field in inverse-Lapace space. The
                presentation time of each stimulus.
            tau_max: float (default = 20)
                The center of the LAST receptive field in inverse-Lapace space. The
                presentation time of each stimulus.
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
        
        self.in_features = in_features
        self.T_every = T_every
        self._T_full_ind = slice(None, None, T_every)
        self.alpha = alpha
        self.g = g

        self._type = ttype
        
        self.lap = Laplace(in_features=in_features, tau_min=tau_min, tau_max=tau_max, 
                           k=k, alpha=alpha, ntau=ntau, ttype=ttype)
        self._Linvk = _calc_Linvk(self.lap.s, k, ttype).unsqueeze(0)
        
        self.tau_star = self.lap.tau_star[slice(k, -k, T_every)]
        
        self._subset_tau_star = (self.lap.tau_star[slice(k, 
                                                         -k, 
                                                         T_every)].unsqueeze(1) ** g).repeat(1, in_features).unsqueeze(0)
        
        
    def reset(self):
        self.lap.reset()
    
    def extra_repr(self):
        s = "out_shape=(1, sequence_len, {tau_star.shape[0]}, {in_features})"
        s = s.format(**self.__dict__)
        return s    
        
    def forward(self, inp, dur=None, alpha=None):
        """
        Takes in a t state and updates with item
        This allows for calculation of little t without storing into
            `self._t`
        Returns new little t state
        """        
        
        # This returns (1, seq, s, features)
        x = self.lap(inp, dur, alpha)
        
        x = torch.bmm(self._Linvk.repeat(x.shape[0], 1, 1),
                      x)[:, self._T_full_ind, :]*self._subset_tau_star.repeat(x.shape[0], 1, 1)


        return x
