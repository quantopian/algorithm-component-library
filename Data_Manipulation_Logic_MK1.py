# import necessary packages
import numpy as np
import pandas as pd
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor
from quantopian.research import run_pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
fundamentals = init_fundamentals()

# function to filter out unwanted values in the scores. this is the filter described in the whitepaper
def filter_fn(x):
    if x <= -10:
        x = -10.0
    elif x >= 10:
        x = 10.0
    return x   

# take data pull and standardize all the values
def standard_frame_compute(df):
    
    # clean dataset
    df = df.replace([np.inf,-np.inf],np.nan)
    df = df.dropna()
    
    # need standardization params from synthetic S&P500
    df_SPY = df.sort(columns='SPY Proxy', ascending=False)
    
    # create separate dataframe for SPY just to get standardization values
    df_SPY = df_SPY.head(500)
    
    # create a dataframe to store the mean and stdev values from each column
    # create a list of column names
    column_list = ['data set','mean', 'stdev']
    SPY_stats = pd.DataFrame(columns=column_list)

    # create pandas dataframe to be used to standardize the universe
    index_no = 0
    for column in df_SPY:
        data = {'data set' :  df_SPY[column].name,
                'mean' : df_SPY[column].mean(),
                'stdev' : df_SPY[column].std()}
        iter_frame = pd.DataFrame(data, index=[index_no])
        SPY_stats = SPY_stats.append(iter_frame)
        index_no += 1
        
    print SPY_stats

    # SPY_stats contains the mean and stdev from the S&P500. Now apply to rest of the data
    # df_standard is like the original df, but this time all values are standardized
    df_standard = pd.DataFrame()
    for i, row in SPY_stats.iterrows():
        if row['data set'] != 'SPY Proxy':
            col = pd.Series(data=((df[row['data set']]-row['mean'])/row['stdev']), name=row['data set'])
            
             # filter between +10 and -10 stdevs  as described in whitepaper
            col = col.apply(lambda x: (filter_fn(x)/(float(len(df_SPY.count(axis=0))-1))))
            
            # add standardized and filtered dataset to the final df
            df_standard = pd.concat([df_standard,col], axis=1)  
            
    return df_standard

# take in a dataframe, return composite score
def composite_score(df):
    
    # sum up transformed data
    df_composite = pd.Series(data=df.sum(axis=1))
    
    # sort descending
    df_composite.sort(ascending=False)
    
    return df_composite


# this function returns a pipeline that downloads all data necessary for the algo
def Data_Pull():
    
    # create the piepline for the data pull
    Data_Pipe = Pipeline()
    
    """
        ADD COMPOSITE FACTORS with Data_Pipe.add(_CUSTOMFACTOR_) HERE
    """
    return Data_Pipe

"""COMPOSITE FACTORS AND DATA PULL LOGIC NEEDED"""