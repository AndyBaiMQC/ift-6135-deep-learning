"""
Template for Question 2 of hwk3.
@author: Samuel Lavoie
"""
import torch
import q2_sampler
import q2_model


def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    # the regularization term is {max(0, |f(x) - f(y)|/||x-y|| - 1)}^2
    # Equation 6 of the paper https://arxiv.org/pdf/1709.08894.pdf
    # penalty was applied to points randomly sampled on the line 
    # between the training sample x and the generated sample y. 
    batch_size = x.size()[0]
    lambdas = torch.rand(batch_size, 1)
    xhat = torch.autograd.Variable(lambdas * x + (1 - lambdas * y), requires_grad=True)
    xhat.retain_grad()

    f_xhat = critic(xhat)
    grad_f_xhat = torch.autograd.grad(
        f_xhat, xhat,
        grad_outputs = torch.ones_like(f_xhat),
        create_graph=True, retain_graph=True
    )[0]
    grad_squared = torch.pow(
        torch.max(
            torch.norm(grad_f_xhat, dim=1)-1,
            torch.zeros(batch_size,1)
        ), 2)
    return torch.mean(grad_squared)
    # f_x_y = (critic(x) - critic(y)).norm()
    # x_y_l2 = torch.norm(x-y, p=2)
    # return torch.max(torch.zeros(batch_size, 1), torch.pow(f_x_y / x_y_l2 - 1, 2))

def vf_wasserstein_distance(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    return torch.mean(critic(x)) - torch.mean(critic(y))


def vf_squared_hellinger(x, y, critic):
    """
    Complete me. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Nowazin et al: https://arxiv.org/pdf/1606.00709.pdf
    In other word, x are samples from the distribution P and y are samples from the distribution Q. Please note that the Critic is unbounded. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Squared Hellinger.
    :return: (FloatTensor) - shape: (1,) - Estimate of the Squared Hellinger
    """
    critic_x = critic(x)
    critic_y = critic(y)
    one_minus_x = 1 - (-critic_x).exp()
    one_minus_y = 1 - (-critic_y).exp()
    return torch.mean(one_minus_x) - torch.mean(one_minus_y/ (torch.ones_like(one_minus_y) - one_minus_y))


if __name__ == '__main__':
    # Example of usage of the code provided for answering Q2.5 as well as recommended hyper parameters.
    model = q2_model.Critic(2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler1 = iter(q2_sampler.distribution1(0, 512))
    theta = 0
    sampler2 = iter(q2_sampler.distribution1(theta, 512))
    lambda_reg_lp = 50 # Recommended hyper parameters for the lipschitz regularizer.
