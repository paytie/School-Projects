from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import corner
import emcee
import lmfit

# Getting and sorting the data
data = get_data('data.txt')
zcmb = data[0]  # redshift
mb = data[1]  # observed B-band magnitude
dmb = data[2]  # error of the observed B-band magnitude


# ----------------- NUMBER 1 --------------------
distance_modulus = []
Mb = -19.3
for i in range(len(mb)):
    distance_modulus.append(get_distance_modulus(mb[i], Mb))


plt.scatter(zcmb, distance_modulus, label='Hubble', color='dimgray', marker='.', zorder=1)
plt.errorbar(zcmb, distance_modulus, xerr=None, yerr=dmb, fmt='.', ecolor='dimgray', color='dimgray', alpha=0.5)
plt.xscale('log')
plt.title('Distance Modulus VS Redshift')
plt.suptitle('Problem 1')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance modulus (Mpc)')
plt.show()

# ----------------- NUMBER 2 --------------------

# Cosmological parameter set 1:
q0_1 = get_q0(0.3, 0.7)
LCDM_1_list = []
for i in range(len(zcmb)):
    LCDM_1 = get_mu_LCDM(70, zcmb[i], q0_1)
    LCDM_1_list.append(LCDM_1)

plt.scatter(zcmb, distance_modulus, label='Hubble', color='dimgray', marker='.', zorder=1)
plt.errorbar(zcmb, distance_modulus, xerr=None, yerr=dmb, fmt='.', ecolor='dimgray', color='dimgray', alpha=0.5)

plt.plot(zcmb, LCDM_1_list, label='Omega_M = 0.3, Omega_L = 0.7', color='darkseagreen', zorder=3)

plt.xscale('log')
plt.legend()
plt.title('Distance Modulus VS Redshift')
plt.suptitle('Condition 1')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance modulus (Actual or Theoretical) (Mpc)')
plt.show()

# Cosmological parameter set 2:
q0_2 = get_q0(1.0, 1.0)
LCDM_2_list = []
for i in range(len(zcmb)):
    LCDM_2 = get_mu_LCDM(70, zcmb[i], q0_2)
    LCDM_2_list.append(LCDM_2)


plt.scatter(zcmb, distance_modulus, label='Hubble', color='dimgray', marker='.', zorder=1)
plt.errorbar(zcmb, distance_modulus, xerr=None, yerr=dmb, fmt='.', ecolor='dimgray', color='dimgray', alpha=0.5)

plt.plot(zcmb, LCDM_2_list, label='Omega_M = 1.0, Omega_L = 1.0', color='moccasin', zorder=3)

plt.xscale('log')
plt.legend()
plt.title('Distance Modulus VS Redshift')
plt.suptitle('Condition 2')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance modulus (Actual or Theoretical) (Mpc)')
plt.show()

# Cosmological parameter set 3:
q0_3 = get_q0(0.3, 0)
LCDM_3_list = []
for i in range(len(zcmb)):
    LCDM_3 = get_mu_LCDM(70, zcmb[i], q0_3)
    LCDM_3_list.append(LCDM_3)

plt.scatter(zcmb, distance_modulus, label='Hubble', color='dimgray', marker='.', zorder=1)
plt.errorbar(zcmb, distance_modulus, xerr=None, yerr=dmb, fmt='.', ecolor='dimgray', color='dimgray', alpha=0.5)

plt.plot(zcmb, LCDM_3_list, label='Omega_M = 0.3, Omega_L = 0', color='lightsalmon', zorder=4)

plt.xscale('log')
plt.legend()
plt.title('Distance Modulus VS Redshift')
plt.suptitle('Condition 3')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance modulus (Actual or Theoretical) (Mpc)')
plt.show()


plt.scatter(zcmb, distance_modulus, label='Hubble', color='dimgray', marker='.', zorder=1)
plt.errorbar(zcmb, distance_modulus, xerr=None, yerr=dmb, fmt='.', ecolor='dimgray', color='dimgray', alpha=0.5)

plt.plot(zcmb, LCDM_1_list, label='Omega_M = 0.3, Omega_L = 0.7', color='darkseagreen', zorder=2)

plt.plot(zcmb, LCDM_2_list, label='Omega_M = 1.0, Omega_L = 1.0', color='moccasin', zorder=3)

plt.plot(zcmb, LCDM_3_list, label='Omega_M = 0.3, Omega_L = 0', color='lightsalmon', zorder=4)

plt.xscale('log')
plt.legend()
plt.title('Distance Modulus VS Redshift')
plt.suptitle('Problem 1 and 2')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance modulus (Actual or Theoretical) (Mpc)')
plt.show()

# ----------------- NUMBER 3 --------------------

difference_1 = []
difference_2 = []
difference_3 = []
for i in range(len(mb)):
    difference_1.append(get_diff_distance_modulus(distance_modulus[i], LCDM_1_list[i]))
    difference_2.append(get_diff_distance_modulus(distance_modulus[i], LCDM_2_list[i]))
    difference_3.append(get_diff_distance_modulus(distance_modulus[i], LCDM_3_list[i]))

plt.scatter(zcmb, difference_1, label='Omega_M = 0.3, Omega_L = 0.7', alpha=0.3, color='darkseagreen', zorder=2)
plt.scatter(zcmb, difference_2, label='Omega_M = 1.0, Omega_L = 1.0', alpha=0.3, color='moccasin', zorder=3)
plt.scatter(zcmb, difference_3, label='Omega_M = 0.3, Omega_L = 0', alpha=0.3, color='lightsalmon', zorder=4)
plt.xscale('log')
plt.legend()
plt.title('Difference of the Distance Modulus (mb - mu_LCDM) VS Redshift')
plt.suptitle('Problem 3')
plt.xlabel('Redshift (z)')
plt.ylabel('Difference of the Distance modulus (Actual - Theoretical) (Mpc)')
plt.axhline(y=0, color='black', linestyle='dashed', zorder=1)
plt.show()


# ----------------- NUMBER 4 --------------------
# done in paper

# ----------------- NUMBER 5 --------------------
# done in paper

# ----------------- NUMBER 6 --------------------
# -.-.-.-.-.-.-.-.- TUTORIAL-.-.-.-.-.-.-.-.-.-.-.

print('')
print('----------------- NUMBER 6 --------------------')
# print('-.-.-.-.-.-.-.-.- TUTORIAL-.-.-.-.-.-.-.-.-.-.-.')
# np.random.seed(123)
#
# # Choose the "true" parameters.
# m_true = -0.9594
# b_true = 4.294
# f_true = 0.534
#
# # Generate some synthetic data from the model.
# N = 50
# x = np.sort(10 * np.random.rand(N))
# yerr = 0.1 + 0.5 * np.random.rand(N)
# y = m_true * x + b_true
# y += np.abs(f_true * y) * np.random.randn(N)
# y += yerr * np.random.randn(N)
#
# plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
# x0 = np.linspace(0, 10, 500)
# plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.TUTORIAL-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
# plt.show()
#
#
# A = np.vander(x, 2)
# C = np.diag(yerr * yerr)
# ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
# cov = np.linalg.inv(ATA)
# w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
# print("Least-squares estimates:")
# print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
# print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))
#
# plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
# plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
# plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
# plt.legend(fontsize=14)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.TUTORIAL-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
# plt.show()
#
#
#
# np.random.seed(42)
# nll = lambda *args: -log_likelihood(*args)
# initial = np.array([m_true, b_true, np.log(f_true)]) + 0.1 * np.random.randn(3)
# soln = minimize(nll, initial, args=(x, y, yerr))
# m_ml, b_ml, log_f_ml = soln.x
#
# print("Maximum likelihood estimates:")
# print("m = {0:.3f}".format(m_ml))
# print("b = {0:.3f}".format(b_ml))
# print("f = {0:.3f}".format(np.exp(log_f_ml)))
#
# plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
# plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
# plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
# plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_ml]), ":k", label="ML")
# plt.legend(fontsize=14)
# plt.xlim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.TUTORIAL-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
# plt.show()
#
# pos = soln.x + 1e-4 * np.random.randn(32, 3)
# nwalkers, ndim = pos.shape
#
# sampler = emcee.EnsembleSampler(
#     nwalkers, ndim, log_probability, args=(x, y, yerr)
# )
# sampler.run_mcmc(pos, 5000, progress=True)
#
# fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
# samples = sampler.get_chain()
# labels = ["m", "b", "log(f)"]
# for i in range(ndim):
#     ax = axes[i]
#     ax.plot(samples[:, :, i], "k", alpha=0.3)
#     ax.set_xlim(0, len(samples))
#     ax.set_ylabel(labels[i])
#     ax.yaxis.set_label_coords(-0.1, 0.5)
#
# axes[-1].set_xlabel("step number")
# plt.show()
#
#
# tau = sampler.get_autocorr_time()
# print(tau)
#
# flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
# print(flat_samples.shape)
#
# fig = corner.corner(
#     flat_samples, labels=labels, truths=[m_true, b_true, np.log(f_true)]
# )
# plt.show()


# -.-.-.-.-.-.-.- FIT FOR MODEL-.-.-.-.-.-.-.-.-.-

print('')
print('-.-.-.-.-.-.-.- FIT FOR MODEL-.-.-.-.-.-.-.-.-.-')

truths = (0.3, 0.0, 70)

x = np.array(zcmb)
np.random.seed(0)
y = model_equation(x, *truths)+0.1*np.random.randn(x.size)
yerr = np.array(dmb)


model = lmfit.Model(model_equation)
p = model.make_params()
p.add('OM', value=0.5, min=0, max=1)
p.add('OL', value=0.5, min=0, max=1)
p.add('H0', value=70, min=0, max=1000)
print(p)
result = model.fit(data=y, params=p, x=x, method='Nelder', nan_policy='omit')

report = lmfit.report_fit(result)
print(report)
result.plot()
plt.xscale('log')
plt.show()

emcee_kws = dict(steps=5000, burn=500, thin=20, is_weighted=False,
                 progress=True)
emcee_params = model.make_params()
emcee_params.add('OM', value=result.params['OM'].value, min=0, max=0.5)
emcee_params.add('OL', value=result.params['OL'].value, min=0, max=1)
emcee_params.add('H0', value=result.params['H0'].value, min=0, max=1000)
emcee_params.add('__lnsigma', value=log10(0.1), min=log10(0.001), max=log10(2.0))

result_emcee = model.fit(data=y, x=x, params=emcee_params, method='emcee',
                         nan_policy='omit', fit_kws=emcee_kws)

emcee_report = lmfit.report_fit(result_emcee)
print(emcee_report)

result_emcee.plot()
plt.xscale('log')
plt.show()
result_emcee.plot_fit()
plt.plot(x, model.eval(params=result.params, x=x), '--', label='emcee')
plt.legend()
plt.xscale('log')
plt.show()

emcee_corner = corner.corner(result_emcee.flatchain, labels=result_emcee.var_names,
                             truths=list(result_emcee.params.valuesdict().values()))
plt.show()








