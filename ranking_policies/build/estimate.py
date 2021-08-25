# python 3.7
# file: data.py
"""Functions/classes to load and clean data."""
# Standard Library Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

# Third-part Imports
import lmfit as lm

# Local Imports / PATH changes

# Authorship
__author__ = "Aaron Watt, Larry Karp"
__copyright__ = "Copyright 2021, ACWatt"
__credits__ = ["Aaron Watt", "Larry Karp"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Aaron Watt"
__email__ = "aaron@acwatt.net"
__status__ = "Prototype"


# FUNCTIONS ===================================================================


def fun_rosenbrock(x):
    return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])


def residuals(params, ebar, ebar_lag, t, t_lag):
    rho = params['rho']
    alpha = params['alpha']
    d0 = params['delta0']
    d1 = params['delta1']
    model = rho * ebar_lag + alpha + d0 * t + d1 * t ** 2 - rho * (d0 * t_lag + d1 * t_lag ** 2)
    return ebar - model


def residuals2(params, ebar, ebar_lag, t, t_lag):
    rho = params[0]
    alpha = params[1]
    d0 = params[2]
    d1 = params[3]
    model = rho * ebar_lag + alpha + d0 * t + d1 * t ** 2 - rho * (d0 * t_lag + d1 * t_lag ** 2)
    return ebar - model


def estimation_andy(data):
    """Reproduce Andy's work in Stata"""
    # Estimate Eq. 71 with nonlinear regression
    min_year = 1945
    max_years = 2005
    data = data[data.ebar_lag.notnull()]  # need to remove Nan values before regressing
    # Estimate nonlinear model:
    # ebar = rho*ebar_lag + (1-rho)*B0 + h(t) - rho*h(t-1)
    # define alpha = (1-rho)*B0
    # use NL estimation to find rho, alpha, delta_0, delta_1
    # Solve for B0 using B0 = alpha / (1-rho)
    # Get Std Err of B0 using variance of B0^0.5
    ebar = data.ebar
    ebar_lag = data.ebar_lag
    t = data.time
    t_lag = data.time_lag

    params = lm.Parameters()
    params.add('rho', value=0.1)
    params.add('alpha', value=1)  # (1 - rho) * B0
    params.add('delta0', value=1)
    params.add('delta1', value=1)

    # compare lmfit minimize covariance
    result = lm.minimize(residuals, params, args=(ebar, ebar_lag, t, t_lag))
    rho = result.params['rho'].value
    alpha = result.params['alpha'].value
    d0 = result.params['delta0'].value
    d1 = result.params['delta1'].value
    B0 = alpha / (1 - rho)
    cov = result.covar
    print('cov: \n', cov)
    var = np.sqrt(np.diagonal(cov))
    rho_v = var[0]
    alpha_v = var[1]

    # compare scipy leastsquares covariance
    params_guess = [0.1, 1, 1, 1]  # rho, alpha, delta0, delta1
    result2 = least_squares(residuals2, params_guess, args=(ebar, ebar_lag, t, t_lag), max_nfev=1000)
    J = result2.jac
    cov = np.linalg.inv(J.T.dot(J))
    print('cov: \n', cov)
    var = np.sqrt(np.diagonal(cov))
    rho = result2.x[0]
    rho_v = var[0]
    alpha = result2.x[1]
    alpha_v = var[1]
    # Variance formula from [1] (references at bottom)
    B0_v = alpha ** 2 * rho_v / (1 - rho) ** 4 \
           + alpha_v / (1 - rho) ** 2 \
           + 2 * alpha * np.sqrt(cov[0, 1])

    # The covariance of both methods seem to be very sensitive to convergence parameters (number of iterations...)
    # need to figure out if this is the right covariance matrix for both methods. Perhaps use a simple sinthetic example
    # below and try both methods.


def residual(params, x, data, eps_data):
    amp = params['amp']
    phaseshift = params['phase']
    freq = params['frequency']
    decay = params['decay']
    model = amp * np.sin(x * freq + phaseshift) * np.exp(-x * x * decay)
    return (data - model) / eps_data


# # generate synthetic data with noise
# x = np.linspace(0, 100)
# eps_data = np.random.normal(size=x.size, scale=0.2)
# data = 7.5 * np.sin(x*0.22 + 2.5) * np.exp(-x*x*0.01) + eps_data
#
# params = lm.Parameters()
# params.add('amp', value=10)
# params.add('decay', value=0.007)
# params.add('phase', value=0.2)
# params.add('frequency', value=3.0)
#
# out = lm.minimize(residual, params, args=(x, data, eps_data))
#
# amp = out.params['amp'].value
# freq = out.params['frequency'].value
# phaseshift = out.params['phase'].value
# decay = out.params['decay'].value
#
# prediction = amp * np.sin(x*freq + phaseshift) * np.exp(-x*x*decay)
# plt.figure()
# plt.plot(x, data)
# plt.plot(x, prediction)
# plt.show()


# MAIN ========================================================================
if __name__ == '__main__':
    pass

# REFERENCES
"""
Variance formula for ratio of paramters
https://stats.stackexchange.com/questions/151974/standard-error-of-the-combination-of-estimated-parameters
"""
