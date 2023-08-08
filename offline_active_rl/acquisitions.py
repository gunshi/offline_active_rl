import numpy as np

from scipy import stats
import ipdb
import torch

_eps = 1e-7

def random(mu, a, temperature=1.0):
    ensembles, batch, classes = mu.shape
    return torch.ones((batch), dtype=torch.float64)       

def mu_probs(mu, a, temperature=1.0):
    # ipdb.set_trace()
    ensembles, batch, classes = mu.shape
    mu_probs = torch.softmax(mu, dim=2)

    mu_probs_in_question = torch.gather(mu_probs, 2, a.unsqueeze(0).repeat(ensembles, 1, 1)).squeeze(-1)
    mu_probs_var = mu_probs_in_question.var(0)

    return mu_probs_var.double()

def mu_realadv(mu, a, temperature=1.0):
    ensembles, batch, classes = mu.shape
    mu_probs = torch.softmax(mu, dim=2)

    adv = (mu_probs * mu).sum(dim=-1) # elt wise mult
    mu_in_question = torch.gather(mu, 2, a.unsqueeze(0).repeat(ensembles, 1, 1)).squeeze(-1)

    mu_adv = (mu_in_question - adv).var(0)
    return mu_adv.double()

def mu_adv(mu, a, temperature=1.0):
    ensembles, batch, classes = mu.shape
    mu_in_question = torch.gather(mu, 2, a.unsqueeze(0).repeat(ensembles, 1, 1)).squeeze(-1)

    mu_avg = mu.mean(dim=2)
    mu_min_mu_avg = (mu_in_question - mu_avg).var(0)
    return mu_min_mu_avg.double()


FUNCTIONS = {
    "random": random,
    "mu_adv": mu_adv,   # var in outcome of this action
    "mu_probs": mu_probs,   # var in prob of this action
    "mu_realadv": mu_realadv,
}
