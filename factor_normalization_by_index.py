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
    df: pandas.DataFrame
        full result of Data_Pull

    returns
    -------
    pandas.DataFrame
        standardized Data_Pull results
    """

    # basic clean of dataset to remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # need standardization params from synthetic S&P500
    df_SPY = df.sort(columns='SPY Proxy', ascending=False)

    # create separate dataframe for SPY
    # to store standardization values
    df_SPY = df_SPY.head(500)

    # create a dataframe to store the
    # mean and stdev values from each column
    column_list = ['data set', 'mean', 'stdev']
    SPY_stats = pd.DataFrame(columns=column_list)

    # create pandas dataframe to standardize the universe
    index_no = 0
    for column in df_SPY:
        data = {'data set':  df_SPY[column].name,
                'mean': df_SPY[column].mean(),
                'stdev': df_SPY[column].std()}
        iter_frame = pd.DataFrame(data, index=[index_no])
        SPY_stats = SPY_stats.append(iter_frame)
        index_no += 1

    # loop through each column and use the appropriate
    # values from SPY_stats to standardize
    df_standard = pd.DataFrame()
    for i, row in SPY_stats.iterrows():
        if row['data set'] != 'SPY Proxy':
            col = pd.Series(
                            data=((df[row['data set']] -
                                    row['mean']) /
                                    row['stdev']),
                            name=row['data set'])

            # apply filter
            col = col.apply(
                            lambda x: (filter_fn(x) /
                            (float(len(df_SPY.count(axis=0)) - 1))))

            # add standardized and filtered dataset to the final df
            df_standard = pd.concat([df_standard, col], axis=1)

    return df_standard


def composite_score(df):
    """
    Summarize standardized data in a single number.

    parameters
    ----------
    df: pandas.DataFrame
        standardized results

    returns
    -------
    pandas.Series
        series of summarized, ranked results

    """

    # sum up transformed data
    df_composite = pd.Series(data=df.sum(axis=1))

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
