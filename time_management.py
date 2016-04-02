from zipline.utils import tradingcalendar


def minutes_till_close():
    """
    Returns the number of minutes until the
    market close
    """
    now = get_datetime()
    open_and_closes = tradingcalendar.open_and_closes
    dt = tradingcalendar.canonicalize_datetime(now)
    idx = open_and_closes.index.searchsorted(dt)
    close = open_and_closes.iloc[idx]['market_close']
    return (close - now).seconds / 60


def get_market_close():
    """
    Returns the market close time for the
    current trading day
    """
    closes = tradingcalendar.open_and_closes['market_close']
    today = tradingcalendar.canonicalize_datetime(get_datetime())
    return closes[today]