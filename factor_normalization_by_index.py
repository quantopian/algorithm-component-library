"""
    This is the main  logic documented in
     Andrew Lo and Pankaj Patel's whitepaper:
     "130/30: The New Long Only"
     (https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf)

     Given user-defined CustomFactors, this code
     will standardize these factors using the mean
     and standard deviation of the S&P500 with respect
     to the CustomFactors.

     The algorithm also cleans and amalgamates
     these standardized results into a single
     composite factor, which can be traded
     with a 130/30 strategy.
"""

import numpy as np
import pandas as pd
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor
from quantopian.research import run_pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
fundamentals = init_fundamentals()


class SPY_proxy(CustomFactor):
    """
    Creates a crude approximation of the
    S&P500, which is used as the basis for
    standardization in the algorithm

    """
    inputs = [morningstar.valuation.market_cap]
    window_length = 1

    def compute(self, today, assets, out, mc):
        out[:] = mc[-1]


def filter_fn(x):
    """
    Cleans the standardized dataset so that
    the composite score can be calculated

    """
    if x <= -10:
        x = -10.0
    elif x >= 10:
        x = 10.0
    return x


def standard_frame_compute(df):
    """
    Standardizes the Pipeline API data pull
    using the S&P500's means and standard deviations for
    particular CustomFactors.

    parameters
    ----------
    df: numpy.array
        full result of Data_Pull

    returns
    -------
    numpy.array
        standardized Data_Pull results

    numpy.array
        index of equities
    """

    # basic clean of dataset to remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # need standardization params from synthetic S&P500
    df_SPY = df.sort(columns='SPY Proxy', ascending=False)

    # create separate dataframe for SPY
    # to store standardization values
    df_SPY = df_SPY.head(500)

    # get dataframes into numpy array
    df_SPY = df_SPY.as_matrix()

    # store index values
    index = df.index.values
    df = df.as_matrix()

    df_standard = np.empty(df.shape[0])

    for col_SPY, col_full in zip(df_SPY.T, df.T):

        # summary stats for S&P500
        mu = np.mean(col_SPY)
        sigma = np.std(col_SPY)
        col_standard = np.array(((col_full - mu) / sigma))

        # create vectorized function (lambda equivalent)
        fltr = np.vectorize(filter_fn)
        col_standard = (fltr(col_standard))

        # make range between -10 and 10
        col_standard = (col_standard / df.shape[1])

        # attach calculated values as new row in df_standard
        df_standard = np.vstack((df_standard, col_standard))

    # get rid of first entry (empty scores)
    df_standard = np.delete(df_standard, 0, 0)

    return (df_standard, index)


def composite_score(df, index):
    """
    Summarize standardized data in a single number.

    parameters
    ----------
    df: numpy.array
        standardized results

    index: numpy.array
        index of equities

    returns
    -------
    pandas.Series
        series of summarized, ranked results

    """

    # sum up transformed data
    df_composite = df.sum(axis=0)

    # put into a pandas dataframe and connect numbers
    # to equities via reindexing
    df_composite = pd.Series(data=df_composite, index=index)

    # sort descending
    df_composite.sort(ascending=False)

    return df_composite


def Data_Pull():
    """
    Attach all CustomFactors to the Pipeline

    returns
    -------
    Pipeline (numpy.array)
        An array containing all data
        needed for the algorithm

    """

    # create the pipeline for the data pull
    Data_Pipe = Pipeline()

    # attach SPY proxy
    Data_Pipe.add(SPY_proxy(), 'SPY Proxy')

    """
        ADD COMPOSITE FACTORS with Data_Pipe.add(CUSTOMFACTOR) HERE
    """
    return Data_Pipe
