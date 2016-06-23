"""
The below is a list of many factors. Taken from several sources,
these factors have been forecast to generate alpha signals either
alone or in combination with eachother. These can form the basis
of factors trading algoithms.

NB: Morningstar cash_flow_statement, income_statement, earnings_report are quarterly

NB: Morningstar ratios are avoided, unless they yield ostensibly better data than otherwise

NB: Multiples of 4 are due to the fact that some metrics are
released quarterly and some are released annually. This is a good approximation and the least comutationally
expensive way to calculate these metrics
"""

import numpy as np
import pandas as pd
import talib
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import Latest
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.data.quandl import fred_usdontd156n as libor


class Factors:
    """List of many factors for use in cross-sectional factor algorithms"""

    """TRADITIONAL VALUE"""

    # Price to Sales Ratio (MORNINGSTAR)
    class Price_To_Sales(CustomFactor):
        """
        Price to Sales Ratio:

        Closing price divided by sales per share.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        Low P/S Ratio suggests that an equity cheap
        Differs substantially between sectors
        """
        inputs = [morningstar.valuation_ratios.ps_ratio]
        window_length = 1

        def compute(self, today, assets, out, ps):
            out[:] = ps[-1]

    # Price to Earnings Ratio (MORNINGSTAR)
    class Price_To_Earnings(CustomFactor):
        """
        Price to Earnings Ratio:

        Closing price divided by earnings per share.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        Low P/E Ratio suggests that an equity cheap
        Differs substantially between sectors
        """
        inputs = [morningstar.valuation_ratios.pe_ratio]
        window_length = 1

        def compute(self, today, assets, out, pe):
            out[:] = pe[-1]

    # Price to Diluted Earnings Ratio (MORNINGSTAR)
    class Price_To_Diluted_Earnings(CustomFactor):
        """
        Price to Diluted Earnings Ratio:

        Closing price divided by diluted earnings per share.
        Diluted Earnings include dilutive securities (Options, convertible bonds etc.) 
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        Low P/Diluted Earnings Ratio suggests that equity is cheap
        Differs substantially between sectors
        """
        inputs = [USEquityPricing.close,
                  morningstar.earnings_report.diluted_eps]
        window_length = 1

        def compute(self, today, assets, out, close, deps):
            out[:] = close[-1] / (deps[-1] * 4)

    # Forward Price to Earnings Ratio (MORNINGSTAR)
    class Price_To_Forward_Earnings(CustomFactor):
        """
        Price to Forward Earnings Ratio:

        Closing price divided by projected earnings for next fiscal period.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        Low P/FY1 EPS Ratio suggests that equity is cheap
        Differs substantially between sectors
        """
        inputs = [morningstar.valuation_ratios.forward_pe_ratio]
        window_length = 1

        def compute(self, today, assets, out, fpe):
            out[:] = fpe[-1]

    # Dividend Yield (MORNINGSTAR)
    class Dividend_Yield(CustomFactor):
        """
        Dividend Yield:

        Dividends per share divided by closing price.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High Dividend Yield Ratio suggests that an equity is attractive to an investor
        as the dividends paid out will be a larger proportion of the price they paid for it.
        """
        inputs = [morningstar.valuation_ratios.dividend_yield]
        window_length = 1

        def compute(self, today, assets, out, dy):
            out[:] = dy[-1]

    # Price to Free Cash Flow (MORNINGSTAR)
    class Price_To_Free_Cashflows(CustomFactor):
        """
        Price to Free Cash Flows:

        Closing price divided by free cash flow.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        Low P/ Free Cash Flows suggests that equity is cheap
        Differs substantially between sectors
        """
        inputs = [USEquityPricing.close,
                  morningstar.valuation_ratios.fcf_per_share]
        window_length = 1

        def compute(self, today, assets, out, close, fcf):
            out[:] = close[-1] / fcf[-1]

    # Price to Operating Cash Flow (MORNINGSTAR)
    class Price_To_Operating_Cashflows(CustomFactor):
        """
        Price to Operating Cash Flows:

        Closing price divided by operating cash flow.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        Low P/ Operating Cash Flows suggests that equity is cheap
        Differs substantially between sectors
        """
        inputs = [USEquityPricing.close,
                  morningstar.valuation_ratios.cfo_per_share]
        window_length = 1

        def compute(self, today, assets, out, close, cfo):
            out[:] = close[-1] / cfo[-1]

    # Price to Book Ratio (MORNINGSTAR)
    class Price_To_Book(CustomFactor):
        """
        Price to Book Value:

        Closing price divided by book value.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        Low P/B Ratio suggests that equity is cheap
        Differs substantially between sectors
        """
        inputs = [USEquityPricing.close,
                  morningstar.valuation_ratios.book_value_per_share]
        window_length = 1

        def compute(self, today, assets, out, close, bv):
            out[:] = close[-1] / bv[-1]

    # Free Cash Flow to Total Assets Ratio (MORNINGSTAR)
    class Cashflows_To_Assets(CustomFactor):
        """
        Cash flows to Assets:

        Operating Cash Flows divided by total assets.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High Cash Flows to Assets Ratio suggests that the company has cash for future operations
        """
        inputs = [morningstar.valuation_ratios.cfo_per_share, morningstar.balance_sheet.total_assets,
                  morningstar.valuation.shares_outstanding]
        window_length = 1

        def compute(self, today, assets, out, cfo, tot_assets, so):
            out[:] = cfo[-1] / (tot_assets[-1] / so[-1])

    # Enterprise Value to Free Cash Flow (MORNINGSTAR)
    class EV_To_Cashflows(CustomFactor):
        """
        Enterprise Value to Cash Flows:

        Enterprise Value divided by Free Cash Flows.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        Low EV/FCF suggests that a company has a good amount of money relative to its size readily available
        """
        inputs = [morningstar.valuation.enterprise_value,
                  morningstar.cash_flow_statement.free_cash_flow]
        window_length = 1

        def compute(self, today, assets, out, ev, fcf):
            out[:] = ev[-1] / fcf[-1]

    # EV to EBITDA (MORNINGSTAR)
    class EV_To_EBITDA(CustomFactor):
        """
        Enterprise Value to Earnings Before Interest, Taxes, Deprecation and Amortization (EBITDA):

        Enterprise Value divided by EBITDA.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        Low EV/EBITDA suggests that equity is cheap
        Differs substantially between sectors / companies
        """
        inputs = [morningstar.valuation.enterprise_value,
                  morningstar.income_statement.ebitda]
        window_length = 1

        def compute(self, today, assets, out, ev, ebitda):
            out[:] = ev[-1] / (ebitda[-1] * 4)

    # EBITDA Yield (MORNINGSTAR)
    class EBITDA_Yield(CustomFactor):
        """
        EBITDA Yield:

        EBITDA divided by close price.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High EBITDA Yield suggests that a company is profitable
        """
        inputs = [USEquityPricing.close, morningstar.income_statement.ebitda]
        window_length = 1

        def compute(self, today, assets, out, close, ebitda):
            out[:] = (ebitda[-1] * 4) / close[-1]

    """MOMENTUM"""

    # Percent Above 260-day Low
    class Percent_Above_Low(CustomFactor):
        """
        Percent Above 260-Day Low:

        Percentage increase in close price between today and lowest close price 
        in 260-day lookback window.
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf

        Notes:
        High value suggests momentum
        """
        inputs = [USEquityPricing.close]
        window_length = 260

        def compute(self, today, assets, out, close):

            # array to store values of each security
            secs = []

            for col in close.T:
                # metric for each security
                percent_above = ((col[-1] - min(col)) / min(col)) * 100
                secs.append(percent_above)
            out[:] = secs

    # Percent Below 260-day High
    class Percent_Below_High(CustomFactor):
        """
        Percent Below 260-Day High:

        Percentage decrease in close price between today and highest close price 
        in 260-day lookback window.
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf
        DB QCD (unsure if public?)

        Notes:
        Low value suggests momentum
        """
        inputs = [USEquityPricing.close]
        window_length = 260

        def compute(self, today, assets, out, close):

            # array to store values of each security
            secs = []

            for col in close.T:
                # metric for each security
                percent_below = ((col[-1] - max(col)) / max(col)) * 100
                secs.append(percent_below)
            out[:] = secs

    # 4/52 Price Oscillator
    class Price_Oscillator(CustomFactor):
        """
        4/52-Week Price Oscillator:

        Average close prices over 4-weeks divided by average close prices over 52-weeks all less 1.
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf

        Notes:
        High value suggests momentum
        """
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):

            # array to store values of each security
            secs = []

            for col in close.T:
                # metric for each security
                oscillator = (np.nanmean(col[-20:]) / np.nanmean(col)) - 1
                secs.append(oscillator)
            out[:] = secs

    # Trendline
    class Trendline(CustomFactor):
        inputs = [USEquityPricing.close]
        """
        52-Week Trendline:

        Slope of the linear regression across a 1 year lookback window.
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf

        Notes:
        High value suggests momentum
        Calculated using the MLE of the slope of the regression 
        """
        window_length = 252

        def compute(self, today, assets, out, close):

            # array to store values of each security
            secs = []

            # days elapsed
            days = xrange(self.window_length)

            for col in close.T:
                # metric for each security
                col_cov = np.cov(col, days)
                secs.append(col_cov[0, 1] / col_cov[1, 1])
            out[:] = secs

    # 1-month Price Rate of Change
    class Price_Momentum_1M(CustomFactor):
        """
        1-Month Price Momentum:

        1-month closing price rate of change.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests momentum (shorter term)
        Equivalent to analysis of returns (1-month window)
        """
        inputs = [USEquityPricing.close]
        window_length = 21

        def compute(self, today, assets, out, close):
            out[:] = (close[-1] - close[0]) / close[0]

    # 3-month Price Rate of Change
    class Price_Momentum_3M(CustomFactor):
        """
        3-Month Price Momentum:

        3-month closing price rate of change.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests momentum (shorter term)
        Equivalent to analysis of returns (3-month window)
        """
        inputs = [USEquityPricing.close]
        window_length = 63

        def compute(self, today, assets, out, close):
            out[:] = (close[-1] - close[0]) / close[0]

    # 6-month Price Rate of Change
    class Price_Momentum_6M(CustomFactor):
        """
        6-Month Price Momentum:

        6-month closing price rate of change.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests momentum (medium term)
        Equivalent to analysis of returns (6-month window)
        """
        inputs = [USEquityPricing.close]
        window_length = 126

        def compute(self, today, assets, out, close):
            out[:] = (close[-1] - close[0]) / close[0]

    # 12-month Price Rate of Change
    class Price_Momentum_12M(CustomFactor):
        """
        12-Month Price Momentum:

        12-month closing price rate of change.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests momentum (long term)
        Equivalent to analysis of returns (12-month window)
        """
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):
            out[:] = (close[-1] - close[0]) / close[0]

    # 12-month Price Rate of Change
    class Returns_39W(CustomFactor):
        """
        39-Week Returns:

        Returns over 39-week window.
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf

        Notes:
        High value suggests momentum (long term)
        Equivalent to analysis of price momentum (39-week window)
        """
        inputs = [USEquityPricing.close]
        window_length = 215

        def compute(self, today, assets, out, close):
            out[:] = (close[-1] - close[0]) / close[0]

    # 1-month Mean Reversion
    class Mean_Reversion_1M(CustomFactor):
        """
        1-Month Mean Reversion:

        1-month returns minus 12-month average of monthly returns over standard deviation
        of 12-month average of monthly returns.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests momentum (short term)
        Equivalent to analysis of returns (12-month window)
        """
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):
            ret_1M = (close[-1] - close[-21]) / close[-21]
            ret_1Y_monthly = ((close[-1] - close[0]) / close[0]) / 12.
            out[:] = (ret_1M - np.nanmean(ret_1Y_monthly)) / \
                np.nanstd(ret_1Y_monthly)

    """EFFICIENCY"""

    # Capital Expenditure to Assets (MORNINGSTAR)
    class Capex_To_Assets(CustomFactor):
        """
        Capital Expnditure to Assets:

        Capital Expenditure divided by Total Assets.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests good efficiency, as expenditure is being used to generate more assets
        """
        inputs = [morningstar.cash_flow_statement.capital_expenditure,
                  morningstar.balance_sheet.total_assets]
        window_length = 1

        def compute(self, today, assets, out, capex, tot_assets):
            out[:] = (capex[-1] * 4) / tot_assets[-1]

    # Capital Expenditure to Sales (MORNINGSTAR)
    class Capex_To_Sales(CustomFactor):
        """
        Capital Expnditure to Sales:

        Capital Expenditure divided by Total Revenue.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests good efficiency, as expenditure is being used to generate greater sales figures
        """
        inputs = [morningstar.cash_flow_statement.capital_expenditure,
                  morningstar.income_statement.total_revenue]
        window_length = 1

        def compute(self, today, assets, out, capex, sales):
            out[:] = (capex[-1] * 4) / (sales[-1] * 4)

    # Capital Expenditure to Cashflows (MORNINGSTAR)
    class Capex_To_Cashflows(CustomFactor):
        """
        Capital Expnditure to Cash Flows:

        Capital Expenditure divided by Free Cash Flows.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests good efficiency, as expenditure is being used to generate greater free cash flows
        """
        inputs = [morningstar.cash_flow_statement.capital_expenditure,
                  morningstar.cash_flow_statement.free_cash_flow]
        window_length = 1

        def compute(self, today, assets, out, capex, fcf):
            out[:] = (capex[-1] * 4) / (fcf[-1] * 4)

    # EBIT to Assets (MORNINGSTAR)
    class EBIT_To_Assets(CustomFactor):
        """
        Earnings Before Interest and Taxes (EBIT) to Total Assets:

        EBIT divided by Total Assets.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests good efficiency, as earnings are being used to generate more assets
        """
        inputs = [morningstar.income_statement.ebit,
                  morningstar.balance_sheet.total_assets]
        window_length = 1

        def compute(self, today, assets, out, ebit, tot_assets):
            out[:] = (ebit[-1] * 4) / tot_assets[-1]

    # Operating Expenditure to Assets (MORNINGSTAR)
    class Operating_Cashflows_To_Assets(CustomFactor):
        """
        Operating Cash Flows to Total Assets:

        Operating Cash Flows divided by Total Assets.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests good efficiency, as more cash being used for operations is being used to generate more assets
        """
        inputs = [morningstar.cash_flow_statement.operating_cash_flow,
                  morningstar.balance_sheet.total_assets]
        window_length = 1

        def compute(self, today, assets, out, cfo, tot_assets):
            out[:] = (cfo[-1] * 4) / tot_assets[-1]

    # Retained Earnings to Assets (MORNINGSTAR)
    class Retained_Earnings_To_Assets(CustomFactor):
        """
        Retained Earnings to Total Assets:

        Retained Earnings divided by Total Assets.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests good efficiency, as greater retained earnings is being used to generate more assets
        """
        inputs = [morningstar.balance_sheet.retained_earnings,
                  morningstar.balance_sheet.total_assets]
        window_length = 1

        def compute(self, today, assets, out, ret_earnings, tot_assets):
            out[:] = ret_earnings[-1] / tot_assets[-1]

    """RISK/SIZE"""

    # Market Cap
    class Market_Cap(CustomFactor):
        """
        Market Capitalization:

        Market Capitalization of the company issuing the equity. (Close Price * Shares Outstanding)
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value for large companies, low value for small companies
        In quant finance, normally investment in small companies is preferred, but thsi depends on the strategy
        """
        inputs = [morningstar.valuation.market_cap]
        window_length = 1

        def compute(self, today, assets, out, mc):
            out[:] = mc[-1]

    # Log Market Cap
    class Log_Market_Cap(CustomFactor):
        """
        Natural Logarithm of Market Capitalization:

        Log of Market Cap. log(Close Price * Shares Outstanding)
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf

        Notes:
        High value for large companies, low value for small companies
        Limits the outlier effect of very large companies through log transformation
        """
        inputs = [morningstar.valuation.market_cap]
        window_length = 1

        def compute(self, today, assets, out, mc):
            out[:] = np.log(mc[-1])

    # Log Market Cap Cubed
    class Log_Market_Cap_Cubed(CustomFactor):
        """
        Natural Logarithm of Market Capitalization Cubed:

        Log of Market Cap Cubed.
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf

        Notes:
        High value for large companies, low value for small companies
        Limits the outlier effect of very large companies through log transformation
        """
        inputs = [morningstar.valuation.market_cap]
        window_length = 1

        def compute(self, today, assets, out, mc):
            out[:] = np.log(mc[-1]**3)

    # Downside Risk
    class Downside_Risk(CustomFactor):
        """
        Downside Risk:

        Standard Deviation of 12-month monthly losses
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests high risk of losses
        """
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):

            stdevs = []
            # get monthly closes
            close = close[0::21, :]
            for col in close.T:
                col_ret = ((col - np.roll(col, 1)) / np.roll(col, 1))[1:]
                stdev = np.nanstd(col_ret[col_ret < 0])
                stdevs.append(stdev)
            out[:] = stdevs

    # Index Beta
    class Index_Beta(CustomFactor):
        """
        Index Beta:

        Slope coefficient of 1-year regression of price returns against index returns
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests high market risk
        Slope calculated using regression MLE
        """
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):

            # get index and calculate returns. SPY code is 8554
            benchmark_index = np.where((assets == 8554) == True)[0][0]
            benchmark_close = close[:, benchmark_index]
            benchmark_returns = (
                (benchmark_close - np.roll(benchmark_close, 1)) / np.roll(benchmark_close, 1))[1:]

            betas = []

            # get beta for individual securities using MLE
            for col in close.T:
                col_returns = ((col - np.roll(col, 1)) / np.roll(col, 1))[1:]
                col_cov = np.cov(col_returns, benchmark_returns)
                betas.append(col_cov[0, 1] / col_cov[1, 1])
            out[:] = betas

    # Downside Beta
    class Downside_Beta(CustomFactor):
        """
        Downside Beta:

        Slope coefficient of 1-year regression of price returns against negative index returns
        http://www.ruf.rice.edu/~yxing/downside.pdf

        Notes:
        High value suggests high exposure to the downmarket
        Slope calculated using regression MLE
        """
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):

            # get index and calculate returns. SPY code is 8554
            benchmark_index = np.where((assets == 8554) == True)[0][0]
            benchmark_close = close[:, benchmark_index]
            benchmark_returns = (
                (benchmark_close - np.roll(benchmark_close, 1)) / np.roll(benchmark_close, 1))[1:]

            # days where benchmark is negative
            negative_days = np.argwhere(benchmark_returns < 0).flatten()

            # negative days for benchmark
            bmark_neg_day_returns = [benchmark_returns[i]
                                     for i in negative_days]

            betas = []

            # get beta for individual securities using MLE
            for col in close.T:
                col_returns = ((col - np.roll(col, 1)) / np.roll(col, 1))[1:]
                col_neg_day_returns = [col_returns[i] for i in negative_days]
                col_cov = np.cov(col_neg_day_returns, bmark_neg_day_returns)
                betas.append(col_cov[0, 1] / col_cov[1, 1])
            out[:] = betas

    # 3-month Volatility
    class Vol_3M(CustomFactor):
        """
        3-month Volatility:

        Standard deviation of returns over 3 months
        http://www.morningstar.com/invglossary/historical_volatility.aspx

        Notes:
        High Value suggests that equity price fluctuates wildly
        """

        inputs = [USEquityPricing.close]
        window_length = 63

        def compute(self, today, assets, out, close):

            vols = []
            for col in close.T:
                # compute returns
                log_col_returns = np.log(col / np.roll(col, 1))[1:]
                vols.append(np.nanstd(log_col_returns))
            out[:] = vols

    """GROWTH"""

    # 3-month Sales Growth
    class Sales_Growth_3M(CustomFactor):
        """
        3-month Sales Growth:

        Increase in total sales over 3 months
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value represents large growth (short term)
        """
        inputs = [morningstar.income_statement.total_revenue]
        window_length = 63

        def compute(self, today, assets, out, sales):
            out[:] = ((sales[-1] * 4) - (sales[0]) * 4) / (sales[0] * 4)

    # 12-month Sales Growth
    class Sales_Growth_12M(CustomFactor):
        """
        12-month Sales Growth:

        Increase in total sales over 12 months
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value represents large growth (long term)
        """
        inputs = [morningstar.income_statement.total_revenue]
        window_length = 252

        def compute(self, today, assets, out, sales):
            out[:] = ((sales[-1] * 4) - (sales[0]) * 4) / (sales[0] * 4)

    # 12-month EPS Growth
    class EPS_Growth_12M(CustomFactor):
        """
        12-month Earnings Per Share Growth:

        Increase in EPS over 12 months
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value represents large growth (long term)
        """
        inputs = [morningstar.earnings_report.basic_eps]
        window_length = 252

        def compute(self, today, assets, out, eps):
            out[:] = (eps[-1] - eps[0]) / eps[0]

    """QUALITY"""

    # Asset Turnover
    class Asset_Turnover(CustomFactor):
        """
        Asset Turnover:

        Sales divided by average of year beginning and year end assets
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value represents good financial health
        Varies substantially between sectors
        """
        inputs = [morningstar.income_statement.total_revenue,
                  morningstar.balance_sheet.total_assets]
        window_length = 252

        def compute(self, today, assets, out, sales, tot_assets):

            turnovers = []

            for col in tot_assets.T:
                # average of assets in last two years
                turnovers.append((col[-1] + col[0]) / 2)
            out[:] = (sales[-1] * 4) / turnovers

    # 3-month Asset Growth
    class Asset_Growth_3M(CustomFactor):
        """
        3-month Asset Growth:

        Increase in total assets over 3 months
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value represents good financial health as quantity of assets is increasing
        """
        inputs = [morningstar.balance_sheet.total_assets]
        window_length = 63

        def compute(self, today, assets, out, tot_assets):
            out[:] = (tot_assets[-1] - tot_assets[0]) / tot_assets[0]

    # Current Ratio
    class Current_Ratio(CustomFactor):
        """
        Current Ratio:

        Total current assets divided by total current liabilities
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value represents good financial health as assets are greater than liabilities (>1)
        Morningstar built-in fundamental ratio more accurate than calculated ratio
        """
        inputs = [morningstar.operation_ratios.current_ratio]
        window_length = 1

        def compute(self, today, assets, out, cr):
            out[:] = cr[-1]

    # Asset/Equity Ratio
    class Asset_To_Equity_Ratio(CustomFactor):
        """
        Asset / Equity Ratio

        Total current assets divided by common equity
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that company has taken on substantial debt
        Vaires substantially with industry
        """
        inputs = [morningstar.balance_sheet.total_assets,
                  morningstar.balance_sheet.common_stock_equity]
        window_length = 1

        def compute(self, today, assets, out, tot_assets, equity):
            out[:] = tot_assets[-1] / equity[-1]

    # Interest Coverage
    class Interest_Coverage(CustomFactor):
        """
        Interest Coverage:

        EBIT divided by interest expense
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that company has taken on substantial debt
        Varies substantially with industry
        """
        inputs = [morningstar.income_statement.ebit,
                  morningstar.income_statement.interest_expense]
        window_length = 1

        def compute(self, today, assets, out, ebit, interest_expense):
            out[:] = (ebit[-1] * 4) / (interest_expense[-1] * 4)

    # Debt to Asset Ratio
    class Debt_To_Asset_Ratio(CustomFactor):
        """
        Debt / Asset Ratio:

        Total Debts divided by Total Assets
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that company has taken on substantial debt
        Low value suggests good financial health as assets greater than debt
        Long Term Debt
        """
        inputs = [morningstar.balance_sheet.total_debt,
                  morningstar.balance_sheet.total_assets]
        window_length = 1

        def compute(self, today, assets, out, debt, tot_assets):
            out[:] = debt[-1] / tot_assets[-1]

    # Debt to Equity Ratio
    class Debt_To_Equity_Ratio(CustomFactor):
        """
        Debt / Equity Ratio:

        Total Debts divided by Common Stock Equity
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that company is taking on debts to leverage
        Low value suggests good financial health as little-to-no leveraging 
        Long Term Debt
        """
        inputs = [morningstar.balance_sheet.total_debt,
                  morningstar.balance_sheet.common_stock_equity]
        window_length = 1

        def compute(self, today, assets, out, debt, equity):
            out[:] = debt[-1] / equity[-1]

    # 3-month Net Debt Growth
    class Net_Debt_Growth_3M(CustomFactor):
        """
        3-Month Net Debt Growth:

        Increase in net debt (total debt - cash) over 3 month window
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that a company is acquiring more debt
        Low value suggests good financial health as debt is decreasing 
        Long Term Debt
        """
        inputs = [morningstar.balance_sheet.total_debt,
                  morningstar.balance_sheet.cash_and_cash_equivalents]
        window_length = 92

        def compute(self, today, assets, out, debt, cash):
            out[:] = ((debt[-1] - cash[-1]) - (debt[0] - cash[0])) / \
                (debt[0] - cash[0])

    # Working Capital / Assets
    class Working_Capital_To_Assets(CustomFactor):
        """
        Working Capital / Assets:

        Current Assets less Current liabilities (Working Capital) all divided by Assets
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that company is currently using more money
        than they are saving (holding as assets)
        Low value suggests good financial health as holding more in assets than risking through use
        working_capital more accurate than total_assets and total_liabilities
        """
        inputs = [morningstar.balance_sheet.working_capital,
                  morningstar.balance_sheet.total_assets]
        window_length = 1

        def compute(self, today, assets, out, capital, tot_assets):
            out[:] = capital[-1] / tot_assets[-1]

    # Working Capital / Sales
    class Working_Capital_To_Sales(CustomFactor):
        """
        Working Capital / Sales:

        Current Assets less Current liabilities (Working Capital) all divided by Sales
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that company is currently using more money than taking in through sales
        than they are saving (holding as assets)
        Low value suggests good financial health as sales stronger than money risked through use
        working_capital more accurate than total_assets and total_liabilities
        """
        inputs = [morningstar.balance_sheet.working_capital,
                  morningstar.income_statement.total_revenue]
        window_length = 1

        def compute(self, today, assets, out, capital, sales):
            out[:] = capital[-1] / (sales[-1] * 4)

    """PAYOUT"""

    # Dividend Growth
    class Dividend_Growth(CustomFactor):
        """
        Dividend Growth:

        Growth in dividends observed over a 1-year lookback window
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that rate at which the quantity of dividends paid out is increasing
        Morningstar built-in fundamental better as adjusts inf values
        """
        inputs = [morningstar.earnings_ratios.dps_growth]
        window_length = 1

        def compute(self, today, assets, out, dpsg):
            out[:] = dpsg[-1]

    # Dividend Payout Ratio
    class Dividend_Payout_Ratio(CustomFactor):
        """
        Dividend Payout Ratio:

        Dividends Per Share divided by Earnings Per Share
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that the amount of earnings paid back to equityholders is large
        """
        inputs = [morningstar.earnings_report.dividend_per_share,
                  morningstar.earnings_report.basic_eps]
        window_length = 1

        def compute(self, today, assets, out, dps, eps):
            out[:] = (dps[-1] * 4) / (eps[-1] * 4)

    """PROFITABILITY"""

    # Gross Income Margin
    class Gross_Income_Margin(CustomFactor):
        """
        Gross Income Margin:

        Gross Profit divided by Net Sales
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that the company is generating large profits
        """
        inputs = [morningstar.income_statement.gross_profit,
                  morningstar.income_statement.total_revenue]
        window_length = 1

        def compute(self, today, assets, out, gp, sales):
            out[:] = (gp[-1] * 4) / (sales[-1] * 4)

    # Net Income Margin
    class Net_Income_Margin(CustomFactor):
        """
        Gross Income Margin:

        Gross Profit divided by Net Sales
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that the company is generating large profits
        Ratio used as cleans inf values
        """
        inputs = [morningstar.operation_ratios.net_margin]
        window_length = 1

        def compute(self, today, assets, out, nm):
            out[:] = nm[-1]

    # Return on Total Equity
    class Return_On_Total_Equity(CustomFactor):
        """
        Return on Total Equity:

        Net income divided by average of total shareholder equity
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that the company is generating large profits
        Ratio used as cleans inf values
        """
        inputs = [morningstar.operation_ratios.roe]
        window_length = 1

        def compute(self, today, assets, out, roe):
            out[:] = roe[-1]

    # Return on Total Assets
    class Return_On_Total_Assets(CustomFactor):
        """
        Return on Total Assets:

        Net income divided by average total assets
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that the company is generating large profits
        Ratio used as cleans inf values
        """
        inputs = [morningstar.operation_ratios.roa]
        window_length = 1

        def compute(self, today, assets, out, roa):
            out[:] = roa[-1]

    # Return on Total Equity
    class Return_On_Total_Invest_Capital(CustomFactor):
        """
        Return on Total Invest Capital:

        Net income divided by average total invested capital
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value suggests that the company is generating large profits
        Ratio used as cleans inf values
        """
        inputs = [morningstar.operation_ratios.roic]
        window_length = 1

        def compute(self, today, assets, out, roic):
            out[:] = roic[-1]

    """PRICE REVERSAL"""

    # 10 Day MACD signal line
    class MACD_Signal_10d(CustomFactor):
        """
        10 Day Moving Average Convergence/Divergence (MACD) signal line:

        10 day Exponential Moving Average (EMA) of the difference between
        12-day and 26-day EMA of close price
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf

        Notes:
        High value suggests turning point in positive momentum
        Low value suggests turning point in negative momentum
        """
        inputs = [USEquityPricing.close]
        window_length = 60

        def compute(self, today, assets, out, close):

            sig_lines = []

            for col in close.T:
                # get signal line only
                try:
                    _, signal_line, _ = talib.MACD(col, fastperiod=12,
                                                   slowperiod=26, signalperiod=10)
                    sig_lines.append(signal_line[-1])
                # if error calculating, return NaN
                except:
                    sig_lines.append(np.nan)
            out[:] = sig_lines

    # 20-day Stochastic Oscillator
    class Stochastic_Oscillator(CustomFactor):
        """
        20-day Stochastic Oscillator:

        K = (close price - 5-day low) / (5-day high - 5-day low)
        D = 100 * (average of past 3 K's)
        We use the slow-D period here (the D above)
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf

        Notes:
        High value suggests turning point in positive momentum (expected decrease)
        Low value suggests turning point in negative momentum (expected increase)
        """
        inputs = [USEquityPricing.close,
                  USEquityPricing.high, USEquityPricing.low]
        window_length = 30

        def compute(self, today, assets, out, close, high, low):

            stoch_list = []

            for col_c, col_h, col_l in zip(close.T, high.T, low.T):
                try:
                    _, slowd = talib.STOCH(col_h, col_l, col_c,
                                           fastk_period=5, slowk_period=3, slowk_matype=0,
                                           slowd_period=3, slowd_matype=0)
                    stoch_list.append(slowd[-1])
                # if error calculating
                except:
                    stoch_list.append(np.nan)

            out[:] = stoch_list

    # 5-day Money Flow / Volume
    class Moneyflow_Volume_5d(CustomFactor):
        """
        5-day Money Flow / Volume:

        Numerator: if price today greater than price yesterday, add price * volume, otherwise subtract
        Denominator: prices and volumes multiplied and summed
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf

        Notes:
        High value suggests turning point in positive momentum (expected decrease)
        Low value suggests turning point in negative momentum (expected increase)
        """
        inputs = [USEquityPricing.close, USEquityPricing.volume]
        window_length = 5

        def compute(self, today, assets, out, close, volume):

            mfvs = []

            for col_c, col_v in zip(close.T, volume.T):

                # denominator
                denominator = np.dot(col_c, col_v)

                # numerator
                numerator = 0.
                for n, price in enumerate(col_c.tolist()):
                    if price > col_c[n - 1]:
                        numerator += price * col_v[n]
                    else:
                        numerator -= price * col_v[n]

                mfvs.append(numerator / denominator)
            out[:] = mfvs

    """TECHNICALS"""

    # Merton's Distance to Default
    class Mertons_DD(CustomFactor):
        """
        Merton's Distance to Default:

        Application the BS Formula with assets as S_t and liabilities as the strike
        https://www.bostonfed.org/bankinfo/conevent/slowdown/groppetal.pdf

        Notes:
        Lower value suggests increased default risk of company issuing equity
        """
        inputs = [morningstar.balance_sheet.total_assets,
                  morningstar.balance_sheet.total_liabilities, libor.value, USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, tot_assets, tot_liabilities, r, close):
            mertons = []

            for col_assets, col_liabilities, col_r, col_close in zip(tot_assets.T, tot_liabilities.T,
                                                                     r.T, close.T):
                vol_1y = np.nanstd(col_close)
                numerator = np.log(
                    col_assets[-1] / col_liabilities[-1]) + ((252 * col_r[-1]) - ((vol_1y**2) / 2))
                mertons.append(numerator / vol_1y)

            out[:] = mertons


FACTORS = {
    'Asset Growth 3M': Factors.Asset_Growth_3M,
    'Asset to Equity Ratio': Factors.Asset_To_Equity_Ratio,
    'Asset Turnover': Factors.Asset_Turnover,
    'Capex to Assets': Factors.Capex_To_Assets,
    'Capex to Cashflows': Factors.Capex_To_Cashflows,
    'Capex to Sales': Factors.Capex_To_Sales,
    'Cashflows to Assets': Factors.Cashflows_To_Assets,
    'Current Ratio': Factors.Current_Ratio,
    'Debt to Asset Ratio': Factors.Debt_To_Asset_Ratio,
    'Dividend Growth': Factors.Dividend_Growth,
    'Dividend Payout Ratio': Factors.Dividend_Payout_Ratio,
    'Dividend Yield': Factors.Dividend_Yield,
    'Downside Beta': Factors.Downside_Beta,
    'Downside Risk': Factors.Downside_Risk,
    'EBITDA Yield': Factors.EBITDA_Yield,
    'EBIT to Assets': Factors.EBIT_To_Assets,
    'EPS Growth 12M': Factors.EPS_Growth_12M,
    'EV to Cashflows': Factors.EV_To_Cashflows,
    'EV to EBITDA': Factors.EV_To_EBITDA,
    'Gross Income Margin': Factors.Gross_Income_Margin,
    'Index Beta': Factors.Index_Beta,
    'Interest Coverage': Factors.Interest_Coverage,
    'Log Market Cap': Factors.Log_Market_Cap,
    'Log Market Cap Cubed': Factors.Log_Market_Cap_Cubed,
    'MACD Signal Line': Factors.MACD_Signal_10d,
    'Market Cap': Factors.Market_Cap,
    'Mean Reversion 1M': Factors.Mean_Reversion_1M,
    'Mertons Distance to Default': Factors.Mertons_DD,
    'Moneyflow Volume 5D': Factors.Moneyflow_Volume_5d,
    'Net Debt Growth 3M': Factors.Net_Debt_Growth_3M,
    'Net Income Margin': Factors.Net_Income_Margin,
    'Operating Cashflows to Assets': Factors.Operating_Cashflows_To_Assets,
    'Percent Above Low': Factors.Percent_Above_Low,
    'Percent Below High': Factors.Percent_Below_High,
    'Price Momentum 12M': Factors.Price_Momentum_12M,
    'Price Momentum 1M': Factors.Price_Momentum_1M,
    'Price Momentum 3M': Factors.Price_Momentum_3M,
    'Price Momentum 6M': Factors.Price_Momentum_6M,
    'Price Oscillator': Factors.Price_Oscillator,
    'Price to Book': Factors.Price_To_Book,
    'Price to Diluted Earnings': Factors.Price_To_Diluted_Earnings,
    'Price to Earnings': Factors.Price_To_Earnings,
    'Price to Forward Earnings': Factors.Price_To_Forward_Earnings,
    'Price to Free Cashflows': Factors.Price_To_Free_Cashflows,
    'Price to Operating Cashflows': Factors.Price_To_Operating_Cashflows,
    'Price to Sales': Factors.Price_To_Sales,
    'Retained Earnings to Assets': Factors.Retained_Earnings_To_Assets,
    'Return on Total Assets': Factors.Return_On_Total_Assets,
    'Return on Total Equity': Factors.Return_On_Total_Equity,
    'Return on Invest Capital': Factors.Return_On_Total_Invest_Capital,
    '39 Week Returns': Factors.Returns_39W,
    'Sales Growth 12M': Factors.Sales_Growth_12M,
    'Sales Growth 3M': Factors.Sales_Growth_3M,
    'Stochastic Oscillator': Factors.Stochastic_Oscillator,
    'Trendline': Factors.Trendline,
    'Vol 3M': Factors.Vol_3M,
    'Working Capital to Assets': Factors.Working_Capital_To_Assets,
    'Working Capital to Sales': Factors.Working_Capital_To_Sales,

}
