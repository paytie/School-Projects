from numpy import log10
import numpy as np


def get_data(file_name):
    f = open(file_name, "r")
    first_line = f.readline()  # skipping the first line
    zcdm = []
    mb = []
    dmb = []
    for line in f:
        list = line.split(' ')
        zcdm.append(float(list[1]))
        mb.append(float(list[4]))
        dmb.append(float(list[5]))
    return zcdm, mb, dmb


# ----------------- NUMBER 1 --------------------


def get_distance_modulus(mb, Mb):
    return mb - Mb


# ----------------- NUMBER 2 --------------------


def get_q0(OM, OL):
    return (1/2)*OM - OL


def get_mu_LCDM(H0, z, q0):
    return 43.23 - 5*log10(H0/68) + 5*log10(z) + 1.086*(1-q0)*z


# ----------------- NUMBER 3 --------------------


def get_diff_distance_modulus(mb, LCDM):
    return mb - LCDM


# ----------------- NUMBER 6 --------------------
# -.-.-.-.-.-.-.-.- TUTORIAL-.-.-.-.-.-.-.-.-.-.-.


def double_exp(x, a1, t1, a2, t2):
    return a1*np.exp(-x/t1) + a2*np.exp(-(x-0.1) / t2)



def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_prior(theta):
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf


def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


# -.-.-.-.-.-.-.- FIT FOR MODEL-.-.-.-.-.-.-.-.-.-


def model_equation(x, OM, H0):
    OL = 1 - OM
    return 43.23 - 5*log10(H0/68) + 5*log10(x) + 1.086*(1-((1/2)*OM - OL))*x





