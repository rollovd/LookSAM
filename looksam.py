import torch
from typing import Union

class LookSAM(torch.optim.Optimizer):

    def __init__(self, k, alpha, model, base_optimizer, criterion, rho=0.05, **kwargs):

        """
        LookSAM algorithm: https://arxiv.org/pdf/2203.02714.pdf
        Optimization algorithm that capable of simultaneously minimizing loss and loss sharpness to narrow
        the generalization gap.

        :param k: frequency of SAM's gradient calculation (default: 10)
        :param model: your network
        :param criterion: your loss function
        :param base_optimizer: optimizer module (SGD, Adam, etc...)
        :param alpha: scaling factor for the adaptive ratio (default: 0.7)
        :param rho: radius of the l_p ball (default: 0.1)

        :return: None

        Usage:
            model = YourModel()
            criterion = YourCriterion()
            base_optimizer = YourBaseOptimizer
            optimizer = LookSAM(k=k,
                                alpha=alpha,
                                model=model,
                                base_optimizer=base_optimizer,
                                criterion=criterion,
                                rho=rho,
                                **kwargs)

            ...

            for train_index, data in enumerate(loader):
                loss = criterion(model(samples), targets)
                loss.backward()
                optimizer.step(t=train_index, samples=samples, targets=targets, zero_grad=True)

            ...

        """

        defaults = dict(alpha=alpha, rho=rho, **kwargs)
        self.model = model
        super(LookSAM, self).__init__(self.model.parameters(), defaults)

        self.k = k
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.criterion = criterion

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.criterion = criterion

    @staticmethod
    def normalized(g):
        return g / g.norm(p=2)

    def step(self, t, samples, targets, zero_grad=False):
        if not t % self.k:
            group = self.param_groups[0]
            scale = group['rho'] / (self._grad_norm() + 1e-8)

            for index_p, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                self.state[p]['old_p'] = p.data.clone()
                self.state[f'old_grad_p_{index_p}']['old_grad_p'] = p.grad.clone()

                with torch.no_grad():
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)

            self.criterion(self.model(samples), targets).backward()

        group = self.param_groups[0]
        for index_p, p in enumerate(group['params']):
            if p.grad is None:
                continue
            if not t % self.k:
                old_grad_p = self.state[f'old_grad_p_{index_p}']['old_grad_p']
                g_grad_norm = LookSAM.normalized(old_grad_p)
                g_s_grad_norm = LookSAM.normalized(p.grad)
                self.state[f'gv_{index_p}']['gv'] = torch.sub(p.grad, p.grad.norm(p=2) * torch.sum(
                    g_grad_norm * g_s_grad_norm) * g_grad_norm)

            else:
                with torch.no_grad():
                    gv = self.state[f'gv_{index_p}']['gv']
                    p.grad.add_(self.alpha.to(p) * (p.grad.norm(p=2) / (gv.norm(p=2) + 1e-8) * gv))

            p.data = self.state[p]['old_p']

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )

        return norm
