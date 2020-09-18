# Laplace pytorch Layer
# PyTorch version 0.1.0
# Authors: Brandon G. Jacques and Per B. Sederberg

import torch
from torch import nn as nn
from itertools import accumulate

class Laplace(nn.Module):
    """Laplace Transform of the input signal."""
    
    def __init__(self, in_features, tau_min=1, tau_max=20, k=4,
                 alpha=1.0, ntau=100, ttype=None):
        """The Laplace transform layer, can take in direct input in the form of 
        (1, sequence_len, features, 2) or bit wise input (1, sequence_len, features, 1)

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
            alpha: float (default = 1.0)
                Rate of change in Laplace domain.
            ntau: int (default = 100)
                The desired number of taustars in the final representation, before
                indexing with T_every
    """
        super(Laplace, self).__init__()
        self.in_features = in_features
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.k = k
        self.ntau = ntau
        self.alpha = alpha

        if ttype is None:
            ttype = torch.FloatTensor
        self._type = ttype

        # determine c from range and save it
        self.c = (tau_max/tau_min)**(1./(ntau-1))-1

        # calc tau_star and s
        self.tau_star = tau_min*(1+self.c)**torch.arange(-k, ntau+k).type(ttype)
        self.s = k/self.tau_star

        # pre-calculate e**s
        self._e_s = torch.exp(-1*self.s)
        self.output_size = [None, self.s.shape[0], in_features]
        self.t = torch.zeros(self.s.shape[0], in_features).type(ttype)
        
    def extra_repr(self):
        s = "{in_features}, {tau_min}-{tau_max} with {ntau} ntau, k={k}, c={c:.4f}"
        s = s.format(**self.__dict__)
        return s
        
    def reset(self):
        # reset all to zeros
        # self.t.zero_()
        # PBS/BGJ: Figure out how to zero grad here 
        # I think we might just be creating a new tensor
        self.t = torch.zeros(self.s.shape[0], self.in_features).type(self._type)
        
        
    def forward(self, inp, dur=None, alpha=None):
        """Handles input of (sequence_len, features) or (sequence_len, features, 2)
        """
        self.reset()
        # we will eventually allow for full batches
        if dur is None:
            dur = self.tau_min
        
        if alpha is None:
            alpha = self.alpha
        
        # determine decay taking into account duration and alpha
        e_alph_dur = self._e_s**(dur*alpha)
        e_alph_dur_update = e_alph_dur.unsqueeze(1).repeat(1, self.in_features)
        # first dimension is the length of the batch
        #output_tensor = torch.zeros(inp.shape[0], self.output_size[1],
        #                            self.output_size[2]).type(self._type)
        
        if len(inp.shape) == 3:
            # item should be of shape [1, features, 2]
            # create what tIN should be if they say input at time item[0, :, 1]
            inp = inp.unsqueeze(1).repeat(1, self.s.shape[0], 1, 1)
            exp = -self.s.view(1, -1, 1).repeat(inp.shape[0], 1, self.in_features)*inp[:,:,:,1]
            tIN = inp[:,:,:,0]*torch.exp(exp)
        else:
            # propagate features across s and apply decay (decay should be of shape [len(s)]
            decay = ((1-e_alph_dur)/(1.0*self.s).T).unsqueeze(1)
            inp = inp.unsqueeze(1)
            tIN = torch.matmul(decay, inp)
                
        ## index into the batch dimension that should be 1
        #for i, IN in enumerate(tIN.split(1, dim=0)):
        #    # update t
        #    self.t = e_alph_dur_update*self.t + IN
        #
        #    # set the output
        #    output_tensor[i, :, :] = self.t
        
        # replace for loop with accumulate from itertools
        output_tensor = accumulate([self.t] + list(tIN.split(1, dim=0)), 
                                   lambda t, inp: e_alph_dur_update*t + inp,)
        
        # ignore the starting value we needed to provide
        next(output_tensor)
        
        # concatenate the results
        output_tensor = torch.cat(list(output_tensor), dim=0)
        
        # save out the state of t (the last state)
        self.t = output_tensor[-1]
        
        return output_tensor
    
