"""
QuanTA-Lib

Author: Gil Wassermann

This is an implementation of TA-Lib for the Quantopian platform.
It is several orders of magnitudes faster than the normal, iterative implementation.

For information on the original implemenation, see http://mrjbq7.github.io/ta-lib/funcs.html
"""

import numpy as np
import pandas as pd
import math
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor, EWMA


class Quantalib:

	"""
	HELPER FUNCTIONS
	"""

	def plus_dm_helper(high, low):
		"""
		Returns positive directional movement. Abstracted for use with more complex factors

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI

		Parameters
		----------
		high : np.array
			matrix of high prices
		low : np.array
			matrix of low prices

		Returns
		-------
		np.array : matrix of positive directional movement

		"""
		# get daily differences between high prices
    	high_diff = (high - np.roll(high, 1, axis=0))[1:]

    	# get daily differences between low prices
    	low_diff = (np.roll(low, 1, axis=0) - low)[1:]

    	# matrix of positive directional movement
    	return np.where(((high_diff > 0) | (low_diff > 0)) & (high_diff > low_diff), high_diff, 0.)

	def minus_dm_helper(high, low):
		"""
		Returns negative directional movement. Abstracted for use with more complex factors

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI

		Parameters
		----------
		high : np.array
			matrix of high prices
		low : np.array
			matrix of low prices

		Returns
		-------
		np.array : matrix of negative directional movement

		"""
		# get daily differences between high prices
    	high_diff = (high - np.roll(high, 1, axis=0))[1:]

    	# get daily differences between low prices
    	low_diff = (np.roll(low, 1, axis=0) - low)[1:]

    	# matrix of megative directional movement
    	return np.where(((high_diff > 0) | (low_diff > 0)) & (high_diff < low_diff), low_diff, 0.)


    def trange_helper(high, low, close):
		"""
		Returns true range

		http://www.macroption.com/true-range/

		Parameters
		----------
		high : np.array
			matrix of high prices
		low : np.array
			matrix of low prices
		close: np.array
			matrix of close prices

		Returns
		-------
		np.array : matrix of true range

		"""
		# define matrices to be compared
	    close = close[:-1]
	    high = high[1:]
	    low = low[1:]

	    # matrices for comparison
	    high_less_close = high - close
	    close_less_low = close - low
	    high_less_low = high - low

	    # return maximum value for each cel
	    return np.maximum(high_less_close, close_less_low, high_less_low)


	"""
	Techincal Analysis Indicators
	"""


	class AD(CustomFactor):
		"""
		Chaikin Accumulation Distribution Line

		Volume indicator

		**Default Inputs:** USEquityPricing.high, USEquityPricing.low, USEquitypricing.close, USEquityPricing.volume

		**Default Window Length:** 14

		http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:accumulation_distribution_line
		"""	    
	    inputs = [USEquityPricing.close, USEquityPricing.high, USEquityPricing.low, USEquityPricing.volume]
	    window_length = 14
	    
	    def compute(self, today, assets, out, close, high, low, vol):

	        # close location value
	        clv = ((close - low) - (high - close)) / (high - low)
	        ad = clv * vol
	        out[:] =  np.sum(ad, axis=0)


	class ADOSC(CustomFactor):
		"""
		Chaikin Accumulation Distribution Oscillator

		Volume Indicator TODO!!!!!1

		"""


	class ADX(CustomFactor):
		"""
		Average Directional Movement Index

		Momentum indicator. Smoothed DX

		**Default Inputs:** USEquityPricing.high, USEquityPricing.low, USEquitypricing.close

		**Default Window Length:** 29

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI
		"""	    
	    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
	    window_length = 29
	    
	    def compute(self, today, assets, out, high, low, close):

	    	# positive directional index
	        plus_di = 100 * np.cumsum(plus_dm_helper(high, low) / trange_helper(high, low, close), axis=0)

	        # negative directional index
	        minus_di = 100 * np.cumsum(minus_dm_helper(high, low) / trange_helper(high, low, close), axis=0)

	        # full dx with 15 day burn-in period
	        dx_frame = (np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100.)[14:]
	        
	        # 14-day EMA
	        span = 14.
	        decay_rate = 2. / (span + 1.)
	        weights = weights_long = np.full(span, decay_rate, float) ** np.arange(span + 1, 1, -1)

	        # return EMA
	        out[:] = np.average(dx_frame, axis=0, weights=weights)


	def APO(shortperiod=12, longperiod=26):
		"""
		Absolute Price Oscillator

		Momentum indeicator. Difference between EWMAs (exponential weighted moving averages) of 
		short and long periods.

		**Default Inputs:** 12, 26

		**Default Window Length:** 46 (26 + 20-day burn-in)

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo

		Parameters
		----------
		shortperiod : int > 0
			window length for short EWMA
		longperiod : int > 0 (>= shortperiod)
			window length for longer EWMA

		Returns
		-------
		zipline.Factor
		"""
		buffer_window = longperiod + 20

		fast = EWMA.from_span(inputs=[USEquityPricing.close], window_length=buffer_window, span=shortperiod)
		slow = EWMA.from_span(inputs=[USEquityPricing.close], window_length=buffer_window, span=longperiod) 
		return fast - slow


	class BOP(CustomFactor):
		"""
		Balance of Power

		Momentum indicator

		**Default Inputs:** USEquityPricing.close, USEquityPricing.open, USEquityPricing.high, USEquityPricing.low

		**Default Window Length:** 1

		https://www.interactivebrokers.com/en/software/tws/usersguidebook/technicalanalytics/balancepower.htm
	    """
	    inputs = [USEquityPricing.close, USEquityPricing.open, USEquityPricing.high, USEquityPricing.low]
	    window_length = 1
	    
	    def compute(self, today, assets, out, close, open, high, low):
	        out[:] = (close - open) / (high - low)


    class CMO(CustomFactor):
		"""
		Chande Momentum Oscillator

		Momentum indicator. Descriptor of overought/oversold conditions

		**Default Inputs:** USEquityPricing.close

		**Default Window Length:** 15 (14 + 1-day for differences)

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo
	    """
	    inputs=[USEquityPricing.close]
	    window_length = 15
	    
	    def compute(self, today, assets, out, close):
	        
	    	# daily differences in close prices
	        close_diff = (close - np.roll(close, 1, axis=0))[1:]

	        # close differences on up days
	        su = np.sum(np.where(close_diff > 0, close_diff, 0), axis=0)

	        # absolute value of close differences on down days
	        sd = np.abs(np.sum(np.where(close_diff < 0, close_diff, 0), axis=0))

	        # CMO
	        out[:] = 100 * (su - sd) / (su + sd)


	class DX(CustomFactor):
		"""
		Directional Movement Index

		Momentum indicator

		**Default Inputs:** USEquityPricing.high, USEquityPricing.low, USEquitypricing.close

		**Default Window Length:** 15

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI
		"""	    
	    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
	    window_length = 15
	    
	    def compute(self, today, assets, out, high, low, close):

	    	# positive directional index
	        plus_di = 100 * np.sum(plus_dm_helper(high, low) / (trange_helper(high, low, close)), axis=0)

	        # negative directional index
	        minus_di = 100 * np.sum(minus_dm_helper(high, low) / (trange_helper(high, low, close)), axis=0)

	        # DX
	        out[:] = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100.


	class LINEARREG_SLOPE(CustomFactor):
		"""
		Slope of Trendline

		Momentum indicator.

		**Default Inputs:**  USEquitypricing.close

		**Default Window Length:** 14

		http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/06/lecture-06.pdf
		"""	    
	    inputs=[USEquityPricing.close]
	    window_length=14

	    # using MLE for speed
	    def compute(self, today, assets, out, close):
	        
	        # prepare X matrix (x_is - x_bar)
	        X = range(self.window_length)
	        X_bar = np.nanmean(X)
	        X_vector = X - X_bar
	        X_matrix = np.tile(X_vector, (len(close.T), 1)).T
	        
	        # prepare Y matrix (y_is - y_bar)
	        Y_bar = np.nanmean(close, axis=0)
	        Y_bars = np.tile(Y_bar, (self.window_length, 1))
	        Y_matrix = close - Y_bars

	        # prepare variance of X
	        X_var = np.nanvar(X)
	        
	        # multiply X matrix an Y matrix and sum (dot product)
	        # then divide by variance of X
	        # this gives the MLE of Beta
	        out[:] = (np.sum((X_matrix * Y_matrix), axis=0) / X_var) / (self.window_length)


	class LINEARREG_INTERCEPT(CustomFactor):
		"""
		Intercept of Trendline

		**Default Inputs:**  USEquitypricing.close

		**Default Window Length:** 14

		http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/06/lecture-06.pdf
		"""
	    inputs=[USEquityPricing.close]
	    window_length=14

	    # using MLE
	    def compute(self, today, assets, out, close):
	        
	        # prepare X matrix (x_is - x_bar)
	        X = range(self.window_length)
	        X_bar = np.nanmean(X)
	        X_vector = X - X_bar
	        X_matrix = np.tile(X_vector, (len(close.T), 1)).T
	        
	        # prepare Y vectors (y_is - y_bar)
	        Y_bar = np.nanmean(close, axis=0)
	        Y_bars = np.tile(Y_bar, (self.window_length, 1))
	        Y_matrix = close - Y_bars
	        
	        # multiply X matrix an Y matrix and sum (dot product)
	        # then divide by variance of X
	        # this gives the MLE of Beta
	        betas = (np.sum((X_matrix * Y_matrix), axis=0) / X_var) / (self.window_length)

	        # prepare variance of X
	        X_var = np.nanvar(X)
	        
	        # now use to get to MLE of alpha
	        out[:] = Y_bar - (betas * X_bar)
        

	class MAX(CustomFactor):
		"""
		Maximum value for each column of a dataset

		**Default Inputs:**  None

		**Default Window Length:** None
		"""
	    def compute(self, today, assets, out, data):
	    	# get vector of maximums
	        out[:] = np.nanmax(data, axis=0)


	class MAXINDEX(CustomFactor):
		"""
		Index of maximum value for each column of a dataset

		**Default Inputs:**  None

		**Default Window Length:** None
		"""
	    def compute(self, today, assets, out, data):
	    	# get vector of indices
	        out[:] = np.argmax(data, axis=0)


	class MEDPRICE(CustomFactor):
		"""
		Mean of a day's high and low prices

		**Default Inputs:**  USEquityPricing.high, USEquityPricing.low

		**Default Window Length:** 1

		http://www.fmlabs.com/reference/default.htm?url=MedianPrices.htm
		"""	    
	    inputs = [USEquityPricing.high, USEquityPricing.low]
	    window_length = 1
	    
	    def compute(self, today, assets, out, high, low):
	        out[:] = (high + low) / 2.


	class MFI(CustomFactor):
		"""
		Money Flow Index

		Volume Indicator

		**Default Inputs:**  USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume

		**Default Window Length:** 15 (14 + 1-day for difference in prices)

		http://www.fmlabs.com/reference/default.htm?url=MoneyFlowIndex.htm
		"""	    
	    
	    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume]
	    window_length = 15
	    
	    def compute(self, today, assets, out, high, low, close, vol):

	    	# calculate typical price
	        typical_price = (high + low + close) / 3.

	        # calculate money flow of typical price
	        money_flow = typical_price * vol
	        
	        # get differences in daily typical prices
	        tprice_diff = (typical_price - np.roll(typical_price, 1, axis=0))[1:]

	        # create masked arrays for positive and negative money flow
	        pos_money_flow = np.ma.masked_array(money_flow[1:], tprice_diff < 0, fill_value = 0.)
	        neg_money_flow = np.ma.masked_array(money_flow[1:], tprice_diff > 0, fill_value = 0.)
	        
	        # calculate money ratio
	        money_ratio = np.sum(pos_money_flow, axis=0) / np.sum(neg_money_flow, axis=0)

	        # MFI
	        out[:] = 100. - (100. / (1. + money_ratio))


	class MIDPOINT(CustomFactor):
		"""
		Average of maximum and minimum column values in a dataset

		**Default Inputs:**  None

		**Default Window Length:** None
		"""
	    def compute(self, today, assets, out, data):
	        out[:] = (np.nanmax(data, axis=0) + np.nanmin(data, axis=0)) / 2.


	class MIN(CustomFactor):
		"""
		Minimum column values in a dataset

		**Default Inputs:**  None

		**Default Window Length:** None
		"""
	    def compute(self, today, assets, out, data):
	        out[:] = np.nanmin(data, axis=0)


	class MININDEX(CustomFactor):
		"""
		Index of minimum column values in a dataset

		**Default Inputs:**  None

		**Default Window Length:** None
		"""
	    def compute(self, today, assets, out, data):
	        out[:] = np.argmin(data, axis=0)


	class MINUS_DI(CustomFactor):
		"""
		Negative directional indicator

		Momentum indicator

		**Default Inputs:** USEquityPricing.high, USEquityPricing.low, USEquityPricing.close

		**Default Window Length:** 15

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI
		"""	
	    
	    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
	    window_length = 15
	    
	    def compute(self, today, assets, out, high, low, close):
	            out[:] = 100 * np.sum(minus_dm_helper(high, low), axis=0) / np.sum(trange_helper(high, low, close), axis=0)


	class MINUS_DM(CustomFactor):
		"""
		Negative directional movement

		Momentum indicator

		**Default Inputs:** USEquityPricing.high, USEquityPricing.low

		**Default Window Length:** 15

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI
		"""	
	    
	    inputs = [USEquityPricing.high, USEquityPricing.low]
	    window_length = 15
	    
	    def compute(self, today, assets, out, high, low):
	            out[:] = np.sum(minus_dm_helper(high, low), axis=0)


	class OBV(CustomFactor):
	    """
	    NOT EXACT MATCH YET. TODO
	    """


	class PLUS_DI(CustomFactor):
		"""
		Positive directional indicator

		Momentum indicator

		**Default Inputs:** USEquityPricing.high, USEquityPricing.low, USEquityPricing.close

		**Default Window Length:** 15

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI
		"""	
	    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
	    window_length = 15
	    
	    def compute(self, today, assets, out, high, low, close):
	            out[:] = 100 * np.sum(plus_dm_helper(high, low), axis=0) / np.sum(trange_helper(high, low, close), axis=0)


	class PLUS_DM(CustomFactor):
		"""
		Positive directional movement

		Momentum indicator

		**Default Inputs:** USEquityPricing.high, USEquityPricing.low

		**Default Window Length:** 15

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/DMI
		"""	
	    inputs = [USEquityPricing.high, USEquityPricing.low]
	    window_length = 15
	    
	    def compute(self, today, assets, out, high, low):
	            out[:] = np.sum(plus_dm_helper(high, low), axis=0)


	def PPO(fast_period=12, slow_period=26):
		"""
		Function to produce CustomFactor of Percentage Price Oscillator
		Called in same way as a normal class, but used in order to give variable
		fast- and slow- periods

		Parameters
		----------
		fast_period : int > 0
			shorter period moving average 
		slow_period : int > 0 (> fast_period)
			longer period moving average

		Returns
		-------
		zipline.CustomFactor : Percentage Price Oscillator factor

		**Default Inputs:**  12, 26
	
		http://www.investopedia.com/terms/p/ppo.asp
		"""  
	    class PPO_(CustomFactor):
	        
	        inputs=[USEquityPricing.close]
	        window_length = slow_period
	        
	        def compute(self, today, assets, out, close):
	            slowMA = np.mean(close, axis=0)
	            fastMA = np.mean(close[-fast_period:], axis=0)
	            out[:] = ((fastMA - slowMA) / slowMA) * 100.

	    return PPO_()


	class STDDEV(CustomFactor):
		"""
		Standard Deviations of columns in a dataset

		**Default Inputs:**  None

		**Default Window Length:** None
		"""
	    def compute(self, today, assets, out, data):
	        out[:] = np.nanstd(data, axis=0)


	class TRANGE(CustomFactor):
		"""
		True Range 

		**Default Inputs:**  USEquityPricing.high, USEquityPricing.low, USEquityPricing.close

		**Default Window Length:** 2

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/atr
		"""    
	    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
	    window_length = 2
	    
	    def compute(self, today, assets, out, high, low, close):
	        out[:] = np.nanmax([(high[-1] - close[0]), (close[0] - low[-1]), (high[-1] - low[-1])], axis=0)


	class TYPPRICE(CustomFactor):
		"""
		Typical Price 

		**Default Inputs:**  USEquityPricing.high, USEquityPricing.low, USEquityPricing.close

		**Default Window Length:** 1

		https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/typical-price
		"""    
	    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
	    window_length = 1
	    
	    def compute(self, today, assets, out, high, low, close):
	        out[:] = (high + low + close) / 3.


	class VAR(CustomFactor):
		"""
		Variances of columns in a dataset

		**Default Inputs:**  None

		**Default Window Length:** None
		"""
	    def compute(self, today, assets, out, data):
	        out[:] = np.nanvar(data, axis=0)
