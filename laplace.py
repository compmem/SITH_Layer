# Laplace pytorch Layer
# PyTorch version 0.1.0
# Authors: Brandon G. Jacques and Per B. Sederberg

import torch
from torch import nn as nn


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
        self._in_features = in_features
        self._tau_min = tau_min
        self._tau_max = tau_max
        self._k = k
        self._ntau = ntau
        self._alpha = alpha

        if ttype is None:
            ttype = torch.FloatTensor
        self._type = ttype

        # determine c from range and save it
        c = (tau_max/tau_min)**(1./(ntau-1))-1
        self._c = c

        # calc tau_star and s
        self._tau_star = tau_min*(1+c)**torch.arange(-k, ntau+k).type(self._type)
        self.s = k/self._tau_star

        # pre-calculate e**s
        self.e_s = torch.exp(-1*self.s)
        self._output_size = [None, self.s.shape[0], self._in_features]
        self.t = torch.zeros(self.s.shape[0], self._in_features).type(self._type)
        
        
    def reset(self):
        # reset all to zeros
        # self.t.zero_()
        # PBS/BGJ: Figure out how to zero grad here 
        # I think we might just be creating a new tensor
        self.t = torch.zeros(self.s.shape[0], self._in_features).type(self._type)
        
        
    def forward(self, inp, dur=None, alpha=None):
        """Handles input of (sequence_len, features) or (sequence_len, features, 2)
        """

        # we will eventually allow for full batches
        if dur is None:
            dur = self._tau_min
        
        if alpha is None:
            alpha = self._alpha
        
        # determine decay taking into account duration and alpha
        e_alph_dur = self.e_s**(dur*alpha)

        # first dimension is the length of the batch
        output_tensor = torch.zeros(inp.shape[0], self._output_size[1],
                                    self._output_size[2]).type(self._type)

        # index into the batch dimension that should be 1
        for i, item in enumerate(inp.split(1, dim=0)):
            # At this point, item should be of size (in_features)
            
            if len(item.shape) == 3:
                # item should be of shape [1, features, 2]
                # create what tIN should be if they say input at time item[0, :, 1]
                tIN = item[0,:,0]*(torch.exp(-self.s.view(-1, 1).repeat(1, self._in_features)*item[0,:,1]))
            else:
                # propagate features across s and apply decay (decay should be of shape [len(s)]
                decay = (1-e_alph_dur)/(1.0*self.s).T
                
                # At this point item should be (1, features) VERY IMPORTANT
                tIN = decay.unsqueeze(1).mm(item)

            # update t
            self.t = e_alph_dur.unsqueeze(1).repeat(1, self._in_features)*self.t + tIN

            # set the output
            output_tensor[i, :, :] = self.t

        return output_tensor
    
