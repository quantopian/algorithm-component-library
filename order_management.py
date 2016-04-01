import numpy as np
import pandas as pd


def last_trade(context, data, sid):
    """
    Returns the last price for a given asset.
    This method is robust to times when the
    asset is not in the data object because it did
    not trade during the period

    parameters
    ----------

    context:
        TradingAlgorithm context object
        passed to handle_data

    data : zipline.protocol.BarData
        data object passed to handle_data

    sid : zipline Asset
        the sid price

    """
    try:
        return data[sid].price
    except KeyError:
        return context.portfolio.positions[sid].last_sale_price



def get_current_holdings(context, data):
    """
    Returns current portfolio holdings
    i.e. number of shares/contracts held of
    each asset.

    parameters
    ----------

    context :
        TradingAlgorithm context object
        passed to handle_data

    data : zipline.protocol.BarData
        data object passed to handle_data

    returns
    -------
    pandas.Series
        contracts held of each asset

    """
    positions = context.portfolio.positions
    return pd.Series({stock: pos.amount
                      for stock, pos in positions.iteritems()})



def get_current_allocations(context, data):
    """
    parameters
    ----------

    context :
        TradingAlgorithm context object
        passed to handle_data

    data : zipline.protocol.BarData
        data object passed to handle_data

    returns
    -------
    pandas.Series


    """
    holdings = get_current_holdings(context, data)
    prices = pd.Series({sid: last_trade(context, data, sid)
                        for sid in holdings.index})
    return prices * holdings / context.portfolio.portfolio_value

