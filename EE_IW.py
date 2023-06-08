# -*- coding: utf-8 -*-

r"""
The Exponentiated Exponential-Inverse Weibull (EE-IW) Model:
Theory and Application to COVID-19 Data in Saudi Arabia.

This file implements the EE-IW distribution.

** Author **
 Elvis Han Cui, Department of Biostatistics, UCLA.

.. [BS2022] M. Badr and G. Sobahi, "The Exponentiated Exponential-
    Inverse Weibull Model: Theory and Application to COVID-19 Data
     in Saudi Arabia,"
    Journal of Mathematics, 2022, https://doi.org/10.1155/2022/8521026.
"""

import numpy as np
import pyswarms as ps

class EE_IW:
    """
    This class computes the Exponentiated Exponential Inverse Weibull (EE-IW) model.
    Density, distribution function, quantile function and random numbers generator
    for the EE-IW Distribution.

    The first parameter (``a``) keyword specifies the IW model.
    The second parameter (``b``) keyword specifies the exponential model.
    The third parameter (``c``) keyword specifies the exponent term.
    """

    def __init__(self, para):
        self.a, self.b, self.c = para
        self.data = None

    def quantile(self, q):
        x = (- 1 / self.c * np.log(1 - (1 - q) ** (1 / self.b))) ** (- 1 / self.a)

        return x

    def cdf(self, x):
        F = 1 - (1 - (np.exp(- self.c * x ** (-self.a)))) ** self.b

        return F

    def pdf(self, x):
        f = self.a * self.b * self.c * x ** (- self.a - 1) * \
            (np.exp(- self.c * x ** (-self.a))) * (1 - (np.exp(- self.c * x ** (-self.a)))) ** (self.b - 1)

        return f

    def grad_a(self, data=None):
        if self.data is None:
            return self.l2(data, self.a, self.b, self.c)
        else:
            return self.l2(self.data, self.a, self.b, self.c)

    def grad_c(self, data=None):
        if self.data is None:
            return self.l1(data, self.a, self.b, self.c)
        else:
            return self.l1(self.data, self.a, self.b, self.c)

    def compute_b(self, data=None):
        if self.data is None:
            return self.b_mle(data, self.a, self.c)
        else:
            return self.b_mle(self.data, self.a, self.c)

    def nega_log_likelihood(self, data=None):
        if not data is None:
            n = len(data)
            lik = n * np.log(self.a * self.b * self.c) - (self.a + 1) * sum(np.log(data)) - \
                  self.c * sum(data ** (-self.a)) + (self.b - 1) * sum(np.log(1 - (np.exp(- self.c * data ** (-self.a)))))
        else:
            n = len(self.data)
            lik = n * np.log(self.a * self.b * self.c) - (self.a + 1) * sum(np.log(self.data)) - \
                    self.c * sum(self.data ** (-self.a)) + (self.b - 1) * sum(
                    np.log(1 - (np.exp(- self.c * self.data ** (-self.a)))))

        return  - lik

    def rvs(self, size=None):
        """
        :param size: tuple.
        :return:
        """
        Q = np.random.random(size)
        X = self.quantile(q=Q)

        return X

    # The following are API functions.
    # l1 computes the gradient of logpdf with respect to c.
    # l2 computes the gradient of logpdf with respect to a.
    # b_mle computes the estimation of b given a and c.

    def l1(self, data, a, b, c):
        n = len(data)
        cache = data ** (-a) / (np.exp(c * data ** (-a)) - 1)
        l1 = n / c - sum(data ** (-a)) + (b - 1) * sum(data ** (-a) * cache)

        return l1

    def l2(self, data, a, b, c):
        n = len(data)
        cache = data ** (-a) / (np.exp(c * data ** (-a)) - 1)
        l2 = n / a - sum(np.log(data)) + c * sum(np.log(data) * (data ** (-a))) - \
             c * (b - 1) * sum(np.log(data) * (data ** (-a)) * cache)

        return l2

    def b_mle(self, data, a, c):
        n = len(data)
        b = - n / sum(np.log(1 - np.exp(- c * (data ** (- a)))))

        return b

def EE_IW_quantile(q, a, b, c):
    x = (- 1 / c * np.log(1 - (1 - q) ** (1 / b))) ** (- 1 / a)

    return x

def EE_IW_cdf(x, a, b, c):
    F = 1 - (1 - (np.exp(- c * x ** (-a)))) ** b

    return F

def EE_IW_pdf(x, a, b, c):
    f = a * b * c * x ** (- a - 1) * (np.exp(- c * x ** (-a))) * (1 - (np.exp(- c * x ** (-a)))) ** (b - 1)

    return f

def l1(data, a, b, c):
    n = len(data)
    cache = data ** (-a) / (np.exp(c * data ** (-a)) - 1)
    l1 = n / c - sum(data ** (-a)) + (b - 1) * sum(data ** (-a) * cache)

    return l1

def l2(data, a, b, c):
    n = len(data)
    cache = data ** (-a) / (np.exp(c * data ** (-a)) - 1)
    l2 = n / a - sum(np.log(data)) + c * sum(np.log(data) * (data ** (-a))) - \
         c * (b - 1) * sum(np.log(data) * (data ** (-a)) * cache)

    return l2

def b_mle(data, a, c):
    n = len(data)
    b = - n / sum(np.log(1 - np.exp(- c * (data ** (- a)))))

    return b

def EE_IW_nega_log_likelihood(data, a, b, c):
    n = len(data)
    lik = n * np.log(a * b * c) - (a + 1) * sum(np.log(data)) - \
          c * sum(data ** (-a)) + (b - 1) * sum(np.log(1 - (np.exp(- c * data ** (-a)))))

    return - lik


def EE_IW_PSO(swarm, data):
    """MLE in a EE-IW Model via PSO.

    Parameters
    ----------
    particles : numpy.ndarray
        sets of inputs shape :code:'(n_particles, dimensions)'

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:'(n_particles, )'
    """

    n_particles = swarm.shape[0]
    result = np.zeros(n_particles)
    mod = EE_IW((0.1, 0.1, 0.1))
    mod.data = data

    for i in range(n_particles):
        mod.a, mod.c = swarm[i, :]
        mod.b = mod.compute_b(data=data)

        #ca1 = mod.grad_a(data=data)
        #ca2 = mod.grad_c(data=data)
        #result[i] = np.abs(ca1) + np.abs(ca2)

        result[i] = mod.nega_log_likelihood(data=data)

    return result


def EE_IW_Model(swarm, data):
    """MLE in a EE-IW Model

    Parameters
    ----------
    particles : numpy.ndarray
        sets of inputs shape :code:'(n_particles, dimensions)'

    Returns
    ----------
    numpy.ndarray
        computed cost of size :code:'(n_particles, )'
    """

    n_particles = swarm.shape[0]
    result = np.zeros(n_particles)
    for i in range(n_particles):
        a, c = swarm[i, :]
        b = b_mle(data, a, c)
        ca1 = l1(data, a, b, c)
        ca2 = l2(data, a, b, c)

        result[i] = np.abs(ca1) + np.abs(ca2)
        result[i] = EE_IW_nega_log_likelihood(data, a, b, c)

    return result