from numpy import log10
import numpy as np


def get_data(file_name): # getting the data from the textfile
    f = open(file_name, "r")
    first_line = f.readline()  # skipping the first line
    zcdm = [] # redshift
    mb = [] # magnitude
    dmb = [] # error
    for line in f:
        list = line.split(' ')
        zcdm.append(float(list[1]))
        mb.append(float(list[4]))
        dmb.append(float(list[5]))
    return zcdm, mb, dmb


# ----------------- NUMBER 1 --------------------


def get_distance_modulus(mb, Mb): # finding distance modulus with magnitude
    return mb - Mb


# ----------------- NUMBER 2 --------------------


def get_q0(OM, OL): # (didn't end up being used) finding the deceleration paramter
    return (1/2)*OM - OL


def get_mu_LCDM(H0, z, q0): # (didn't end up being used) finding the mu_LCDM 
    return 43.23 - 5*log10(H0/68) + 5*log10(z) + 1.086*(1-q0)*z


# ----------------- NUMBER 3 --------------------


def get_diff_distance_modulus(mb, LCDM): # finding the residuals 
    return mb - LCDM


# ----------------- NUMBER 6 --------------------
# -.-.-.-.-.-.-.-.- TUTORIAL-.-.-.-.-.-.-.-.-.-.-.
# all of these are copied and pasted from the tutorial mentioned to make sure emcee worked properly

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







