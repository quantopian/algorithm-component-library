
import numpy as np
import scipy
import pandas as pd



def get_reduced_correlation_weights(R):
    """
    Implementation of minimum correlation algorithm.
    ref: http://cssanalytics.com/doc/MCA%20Paper.pdf

    parameters
    ----------
    R : pandas.DataFrame
        Timeseries of asset returns

    returns
    -------
    pandas.Series
    portfolio weights that minimize the correlation
    in the portfolio.

    """
    correlations = R.corr()
    adj_correlations = get_adjusted_cor_matrix(correlations)
    initial_weights = adj_correlations.T.mean()

    ranks = initial_weights.rank()
    ranks /= ranks.sum()

    weights = adj_correlations.dot(ranks)
    weights /= weights.sum()

    return weights


def get_adjusted_cor_matrix(cor):
    """
    Helper function for get_reduced_correlation_weights

    parameters
    ----------
    cor : pandas.DataFrame
        Asset returns correlation matrix

    returns
    -------
    pandas.DataFrame
        adjusted correlation matrix
    """
    values = cor.values.flatten()
    mu = np.mean(values)
    sigma = np.std(values)
    distribution = scipy.stats.norm(mu, sigma)
    return 1 - cor.apply(lambda x: distribution.cdf(x))



def minimum_var(R):
    """
    Minimum Variance Portfolio weights.

    parameters
    ----------

    R : pandas.DataFrame
        Dataframe of asset returns

    returns
    -------
    pandas.Series
        minimum variance weights

    """
    cov_inv = np.linalg.inv(R.cov())
    ones = np.ones(len(cov_inv))
    v = cov_inv.dot(ones)
    w = v / ones.T.dot(v)
    return pd.Series(w, index=R.columns)


def efficient_frontier(R, target_return):
    """
    Efficient Frontier Portfolio weights.
    An EF portfolio can be thought of as the
    portfolio with the minimum risk for a given target return.

    parameters
    ----------

    R : pandas.DataFrame
        asset returns

    target_return : float
        the target return for the portfolio

    returns
    -------
    pandas.Series
        efficient frontier portfolio weights

    """
    c_inv = np.linalg.inv(R.cov())
    ones = np.ones(len(c_inv))
    mu_t = np.array([target_return, 1.0])
    M = np.array([R.mean(), ones]).T
    B = np.dot(M.T, c_inv.dot(M))
    B_inv = np.linalg.inv(B)
    v = np.dot(c_inv, M)
    u = np.dot(B_inv, mu_t)
    w = np.dot(v, u)
    return pd.Series(w, index=R.columns)


def tangent_portfolio(R, rfr=0.0):
    """
    Modern portfolio theory tangency portfolio
    given a risk free rate

    parameters
    ----------

    R : pandas.DataFrame
        asset returns

    rfr : float
        the risk free rate

    returns
    -------
    pandas.Series
        tangent portfolio weights

    """
    c_inv = np.linalg.inv(R.cov())
    mu = R.mean()
    ones = np.ones(len(mu), dtype=float)
    rf = rfr * ones
    t = c_inv.dot(mu - rf) / ones.T.dot(c_inv.dot(mu - rf))
    return pd.Series(t, index=R.columns)
