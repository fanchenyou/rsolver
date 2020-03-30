# from .optimizer import Optimizer, required
import torch
from torch.optim import optimizer
from torch.optim.optimizer import Optimizer, required
import numpy as np

from gutils import unit
from gutils import gproj
from gutils import clip_by_norm
from gutils import xTy
from gutils import gexp
from gutils import gpt
from gutils import gpt2
from gutils import Cayley_loop
from gutils import qr_retraction
from gutils import check_identity
from utils import matrix_norm_one
import random

episilon = 1e-8


class SGDG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'stiefel'. 

        If stiefel is True, the variables will be updated by SGD-G proposed 
        as decorrelated weight matrix.

        If stiefel is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        stiefel (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case stiefel is False 
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case stiefel is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 stiefel=False, omega=0, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        stiefel=stiefel, omega=0, grad_clip=grad_clip)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.addon_vars = {}
        self.step_cnt = 0

        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @staticmethod
    def proj(X, B):
        bx = B.t().mm(X)
        sym = (bx + bx.t()) / 2
        B_proj = B - X.mm(sym)
        return B_proj

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.step_cnt += 1
        # print(self.step_cnt)

        for group in self.param_groups:
            # print('=================')
            # print(group.keys())
            # print('--------------')

            momentum = group['momentum']
            stiefel = group['stiefel']
            gname = group['name']
            beta = group['beta']

            # print(len(group['params']))

            if gname not in self.addon_vars:
                self.addon_vars[gname] = [None] * len(group['params'])

            # for p in group['params']:
            #     print(p.size(),)

            for ix, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                weight_decay = group['weight_decay']
                dampening = group['dampening']
                nesterov = group['nesterov']

                # unity, _ = unit(p.data.view(p.size()[0], -1))
                if stiefel: # and unity.size()[0] <= unity.size()[1]:

                    lr = group['lr'] / np.sqrt(self.step_cnt)
                    # print(lr)
                    # get Euclidean gradient g
                    g = p.grad.data.view(p.size()[0], -1)

                    # get Riemann gradient by projection
                    # see [LiXiao] eq(9,10)
                    X = p.data.view(p.size()[0], -1)  # we use X to denote original parameter p
                    B = g
                    g_proj = self.proj(X, B)

                    if self.addon_vars[gname][ix] is None:
                        I_0 = torch.zeros(g_proj.size(0))
                        II_0 = torch.zeros(g_proj.size(0))
                        r_0 = torch.zeros(g_proj.size(1))
                        rr_0 = torch.zeros(g_proj.size(1))
                        if g.is_cuda:
                            I_0 = I_0.cuda()
                            II_0 = II_0.cuda()
                            r_0 = r_0.cuda()
                            rr_0 = rr_0.cuda()
                        self.addon_vars[gname][ix] = [I_0, r_0, II_0, rr_0]


                    I_t_1, r_t_1, II_t_1, rr_t_1 = self.addon_vars[gname][ix]
                    n = g.size(0)
                    r = g.size(1)
                    ggt = g_proj.mm(g_proj.t())
                    gtg = g_proj.t().mm(g_proj)
                    I_t = beta * I_t_1 + (1 - beta) * torch.diagonal(ggt) / r
                    II_t = torch.max(II_t_1, I_t)
                    r_t = beta * r_t_1 + (1 - beta) * torch.diagonal(gtg) / n
                    rr_t = torch.max(rr_t_1, r_t)

                    # Algorithm 1, Line 8
                    II_t4 = torch.diag(II_t.pow(-0.25))
                    rr_t4 = torch.diag(rr_t.pow(-0.25))
                    DGD = II_t4.mm(g_proj.mm(rr_t4))
                    DGD_proj = X - lr * self.proj(X, DGD)  ##### Is this projecting DGD back to original parameter p?
                    p_new = qr_retraction(DGD_proj)  ##### Is this retraction OK?

                    # X_Plus = X - lr * g_proj
                    # p_new = qr_retraction(X_Plus)

                    p.data.copy_(p_new.view(p.size()))

                    # param_state = self.state[p]
                    # if 'momentum_buffer' not in param_state:
                    #     param_state['momentum_buffer'] = torch.zeros(g.t().size())
                    #     if p.is_cuda:
                    #         param_state['momentum_buffer'] = param_state['momentum_buffer'].cuda()
                    #
                    # V = param_state['momentum_buffer']
                    # V = momentum * V - g.t()
                    # MX = torch.mm(V, unity)
                    # XMX = torch.mm(unity, MX)
                    # XXMX = torch.mm(unity.t(), XMX)
                    # W_hat = MX - 0.5 * XXMX
                    # W = W_hat - W_hat.t()
                    # t = 0.5 * 2 / (matrix_norm_one(W) + episilon)
                    # alpha = min(t, lr)
                    #
                    # p_new = Cayley_loop(unity.t(), W, V, alpha)
                    # V_new = torch.mm(W, unity.t())  # n-by-p
                    # #                     check_identity(p_new.t())
                    # p.data.copy_(p_new.view(p.size()))
                    # V.copy_(V_new)

                else:
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss
