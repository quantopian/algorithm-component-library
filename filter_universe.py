from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.data import morningstar as mstar


def filter_universe():
    """

    Modified from Nathan Wolfe's Algorithm
    https://www.quantopian.com/posts/pipeline-trading-universe-best-practice

    returns
    -------
    zipline.pipeline.Filter
        Filter for the set_screen of a Pipeline to create an
        ideal trading universe

    The filters:
        1. common stock
        2 & 3. not limited partnership - name and database check
        4. database has fundamental data
        5. not over the counter
        6. not when issued
        7. not depository receipts
        8. primary share
        9. high dollar volume
    """

    common_stock = mstar.share_class_reference.security_type.latest.eq('ST00000001')
    not_lp_name = ~mstar.company_reference.standard_name.latest.matches('.* L[\\. ]?P\.?$')
    not_lp_balance_sheet = mstar.balance_sheet.limited_partnership.latest.isnull()
    have_data = mstar.valuation.market_cap.latest.notnull()
    not_otc = ~mstar.share_class_reference.exchange_id.latest.startswith('OTC')
    not_wi = ~mstar.share_class_reference.symbol.latest.endswith('.WI')
    not_depository = ~mstar.share_class_reference.is_depositary_receipt.latest
    primary_share = IsPrimaryShare()

    # Combine the above filters.
    tradable_filter = (common_stock & not_lp_name & not_lp_balance_sheet &
                       have_data & not_otc & not_wi & not_depository & primary_share)

    high_volume_tradable = (AverageDollarVolume(window_length=21,
                                                mask=tradable_filter).percentile_between(70, 100))

    screen = high_volume_tradable

    return screen
