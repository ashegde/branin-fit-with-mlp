import math
import torch

def branin_function(x: torch.Tensor) -> torch.Tensor:
    # input x is (...,2)
    # output is (...,1)
    assert x.shape[-1] == 2, f"input x should be of dimension (...,2)"

    x1bar = x[...,0]*7.5 + 2.5
    x2bar = x[...,1]*7.5 + 7.5
    f = 1.0 * (x2bar - (5.1/(4*math.pi**2))*x1bar**2 + (5 / math.pi)*x1bar - 6)**2
    f = (f + 10 * (1-(1/(8*math.pi)))*torch.cos(x1bar) + 10)/250
    return f