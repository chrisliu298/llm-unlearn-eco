import torch


class ZerothOrderOptimizerScalar:
    def __init__(self, lr, eps, beta, min_beta):
        self.lr = lr
        self.eps = eps
        self.beta = beta
        self.min_beta = min_beta

    def step(self, score_fn, args):
        backward_min = max(self.beta - self.eps, 1e-8)

        forward_score = score_fn(self.beta + self.eps, **args)
        backward_score = score_fn(backward_min, **args)
        grad_estimate = (forward_score - backward_score) / (2 * self.eps)

        # if grad_estimate > 0:
        #     grad_estimate = -grad_estimate

        self.beta = self.beta - self.lr * grad_estimate
        self.beta = max(self.beta, self.min_beta)

        return {
            "beta": self.beta,
            "f_score": forward_score,
            "b_score": backward_score,
            "grad_est": grad_estimate,
        }


class ZerothOrderOptimizerVector:
    def __init__(self, lr, eps, beta, min_beta):
        self.lr = lr
        self.eps = eps
        self.beta = beta
        self.min_beta = min_beta

    def step(self, score_fn, args):
        u = torch.randn_like(self.beta)
        u = u / u.norm()

        forward_score = score_fn(self.beta + self.eps * u, **args)
        backward_score = score_fn(self.beta - self.eps * u, **args)
        grad_estimate = (forward_score - backward_score) / (2 * self.eps) * u

        # if torch.dot(grad_estimate, u) > 0:
        #     grad_estimate = -grad_estimate

        self.beta = self.beta - self.lr * grad_estimate
        self.beta = torch.max(self.beta, torch.tensor(self.min_beta))

        return self.beta, forward_score, backward_score, grad_estimate
