import numpy as np


def shrink_covariance_matrix(x, shrink=None):
		"""
		Shrinks towards constant correlation matrix
		If shrink is specified then this const is used for shrinkage
			
		The notation follows Ledoit and Wolf (2004) 
		http://www.ledoit.net/honey_abstract.htm
		This version: 06/2009
		
		Parameters
		----------
		x : N x N sample covariance matrix of stock returns
		shrink : given shrinkage intensity factor; if none, code calculates
		
		Returns
		-------
		tuple : numpy.ndarray which contains the shrunk covariance matrix
					: float shrinkage intensity factor
		
		"""
		if x is None:
				raise ValueError('No covariance matrix defined')
		
		if isinstance(x, pandas.DataFrame):
				cov = x.as_matrix()
		elif isinstance(x, np.ndarray):
				cov = x
		else:
				raise ValueError('Covariance matrix passed must be numpy.ndarray or pandas.DataFrame')
		
		if shrink is not None:
				shrinkage = shrink

		[t, n] = np.shape(cov)
		meanx = cov.mean(axis=0)
		cov = cov - np.tile(meanx, (t, 1))
		
		sample = (1.0 / t) * np.dot(cov.T, cov)
		
		var = np.diag(sample)
		sqrtvar = np.sqrt(var)

		a = np.tile(sqrtvar, (n, 1))
		rho = (sum(sum(sample / (a * a.T))) - n) / (n*(n-1))
		
		prior = rho * (a * a.T)
		prior[np.eye(t, n)==1] = var
		
		# Frobenius-norm of matrix cov, sqrt(sum(diag(dot(cov.T, cov))))
		c = np.linalg.norm(sample-prior, 'fro')**2
		y = cov**2.0
		p = np.dot((1.0 / t), sum(sum(np.dot(y.T, y))))-sum(sum(sample**2.0))
		rdiag = np.dot((1.0 / t), sum(sum(y**2.0))) - sum(var**2.0)
		v = np.dot((cov**3.0).T, cov) / t - ((var*sample).T)
		v[np.eye(t, n)==1] = 0.0
		roff = sum(sum(v * (a  / a.T)))
		r = rdiag + np.dot(rho, roff)
		
		# compute shrinkage constant
		k = (p - r) / c
		shrinkage = max(0.0, min(1.0, k/t))
		sigma = np.dot(shrinkage, prior) + np.dot((1 - shrinkage), sample)
		
		return sigma, shrinkage
