"""
The below is a list of many factors. Taken from several sources,
these factors have been forecast to generate alpha signals either
alone or in combination with eachother. These can form the basis
of factors trading algoithms.

NB: Morningstar cash_flow_statement, income_statement, earnings_report are quarterly
"""

import numpy as np
import pandas as pd
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import Latest
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.classifiers.morningstar import Sector


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

        def compute(self, today, assets, out, fcf, tot_assets, so):
            out[:] = fcf[-1] / (tot_assets[-1] / so[-1])

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

    # Percent Above 260d Low
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
        """
        52-Week Trendline:

        Slope of the linear regression across a 1 year lookback window.
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf

        Notes:
        High value suggests momentum
        Calculated using the MLE of the slope of the regression
        """
        inputs = [USEquityPricing.close]
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
        window_length = 92

        def compute(self, today, assets, out, sales):
            out[:] = (sales[-1] - sales[0]) / sales[0]

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
            out[:] = (sales[-1] - sales[0]) / sales[0]

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
        12-month Earnings Per Share Growth:

        Increase in EPS over 12 months
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf

        Notes:
        High value represents large growth (long term)
        """
        inputs = [morningstar.income_statement.total_revenue,
                  morningstar.balance_sheet.total_assets]
        window_length = 253

        def compute(self, today, assets, out, sales, tot_assets):

            turnovers = []

            for col in tot_assets.T:
                # average of assets in last two years
                turnovers.append((col[-1] + col[0]) / 2)
            out[:] = sales[-1] / turnovers
