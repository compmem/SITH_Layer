import torch
from math import factorial

# Impulse-based SITH class
class iSITH(torch.nn.Module):
    def __init__(self, tau_min=.1, tau_max=100., buff_max=200, k=50, ntau=50, dt=1, g=0.0,
                 ttype=torch.FloatTensor):
        super(iSITH, self).__init__()
        
        self.k = k
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.ntau = ntau
        self.dt = dt
        self.g = g
        
        self.c = (tau_max/tau_min)**(1./(ntau-1))-1
        
        self.tau_star = tau_min*(1+self.c)**torch.arange(ntau).type(ttype)
        
        self.times = torch.arange(0, buff_max, dt).type(ttype)
        
        A = ((1/self.tau_star)*(k**(k+1)/factorial(k))*(self.tau_star**self.g)).unsqueeze(1)
        self.filters = A*((self.times.unsqueeze(0)/self.tau_star.unsqueeze(1))**(k+1)) * \
                        torch.exp(k*(-self.times.unsqueeze(0)/self.tau_star.unsqueeze(1)))
        self.filters = torch.flip(self.filters, [-1]).unsqueeze(1).unsqueeze(1)
        
    def forward(self, inp):

        assert(len(inp.shape) >= 4)        
        out = torch.conv2d(inp, self.filters, padding=[0, self.filters.shape[-1]])
        
        # note we're scaling the output by both dt and the k/(k+1)
        return out[:, :, :, 1:inp.shape[-1]+1]*self.dt*self.k/(self.k+1)
