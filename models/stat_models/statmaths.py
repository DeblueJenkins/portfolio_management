import numpy as np


def covariance_matrix(data: np.array) -> np.array:
    """
    compute covariance matrix

    :param data: numpy array of data
    :return: covariance matrix
    """
    x = data - data.mean(axis=0)
    return np.dot(x.T, x)/(len(x) - 1)


def mean(data: np.array) -> np.ndarray:
    """
    compute the multivariate mean

    :param data: numpy array of data
    :return: mean vector
    """
    n_assets = np.shape(data)[1]
    return np.mean(data, axis=0).reshape(n_assets, 1)


def cholesky(c: np.array) -> np.ndarray:
    """
    computes the cholesky decomposition: transpose(a) * a = c

    :param c: a correlation matrix as numpy array
    :return: the "square root" of a matrix
    """
    n = np.shape(c)[0]

    a = np.zeros((n, n))

    for i in range(0, n):
        for k in range(0, i+1):
            cross_sum_k = 0
            for j in range(0, k):
                cross_sum_k += np.sum(a[i][j] * a[k][j])

            if i == k:
                a[i][k] = np.sqrt(c[i][i] - cross_sum_k)
            else:
                a[i][k] = (c[i][k] - cross_sum_k)/a[k][k]

    return a


def _gh_stepsize(vP):

    vh = 1e-8*(np.fabs(vP)+1e-8)   # Find stepsize
    vh = np.maximum(vh, 5e-6)      # Don't go too small

    return vh


def gradient_2sided(fun, vP, *args):

    iP = np.size(vP)
    vP = vP.reshape(iP)
    vh = _gh_stepsize(vP)
    mh = np.diag(vh)
    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):         # Find f(x+h), f(x-h)
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    vhr = (vP + vh) - vP       # Check for effective stepsize right
    vhl = vP - (vP - vh)        # Check for effective stepsize left
    vG = (fp - fm) / (vhr + vhl)  # Get central gradient

    return vG


def hessian_2sided(fun, vP, *args):

    iP = np.size(vP,0)
    vP = vP.reshape(iP)    # Ensure vP is 1D-array
    f = fun(vP, *args)
    vh = _gh_stepsize(vP)
    vPh = vP + vh
    vh = vPh - vP

    mh = np.diag(vh)      # Build a  diagonal matrix out of vh

    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    fpp = np.zeros((iP, iP))
    fmm = np.zeros((iP, iP))
    for i in range(iP):
        for j in range(i, iP):
            fpp[i, j] = fun(vP + mh[i] + mh[j], *args)
            fpp[j, i] = fpp[i, j]
            fmm[i, j] = fun(vP - mh[i] - mh[j], *args)
            fmm[j, i] = fmm[i, j]

    vh = vh.reshape((iP, 1))
    mhh = vh   @  vh.T             # mhh= h  h', outer product of h-vector

    mH = np.zeros((iP,iP))
    for i in range(iP):
        for j in range(i,iP):
            mH[i,j] = (fpp[i,j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i, j])/mhh[i, j]/2
            mH[j,i] = mH[i,j]

    return mH


def jacobian_2sided(fun, vP, *args):

    iP = np.size(vP)
    vP = vP.reshape(iP)        # Ensure vP is 1D-array
    vF = fun(vP, *args)        # evaluate function, only to get size
    iN = vF.size
    vh = _gh_stepsize(vP)
    mh = np.diag(vh)        # Build a  diagonal matrix out of h
    mGp = np.zeros((iN, iP))
    mGm = np.zeros((iN, iP))
    for i in range(iP):     # Find f(x+h), f(x-h)
        mGp[:,i] = fun(vP+mh[i], *args)
        mGm[:,i] = fun(vP-mh[i], *args)
    vhr = (vP + vh) - vP    # Check for effective stepsize right
    vhl = vP - (vP - vh)    # Check for effective stepsize left
    mG = (mGp - mGm) / (vhr + vhl)  # Get central jacobian
    return mG


def GetCovML(fnNAvgLnL, vP, iN, *args):

    mH = hessian_2sided(fnNAvgLnL, vP, *args)
    mS2 = np.linalg.inv(mH)
    mS2 = (mS2 + mS2.T)/2       #  Force mS2 to be symmetric

    return mS2


def covariance(theta: np.array, average_likelihood_func: callable, n: int):
    # compute the inverse hessian of the average log likelihood
    h = hessian_2sided(average_likelihood_func, theta)
    cov = np.linalg.inv(h)
    cov = (cov + cov.T) / 2
    return cov / n


def make_covariance_robust(cov, theta: np.array, likelihood_func: callable):
    n = np.shape(cov)[0]
    jac = jacobian_2sided(likelihood_func, theta)
    jac = np.dot(jac.T, jac) / n
    m = np.dot(jac, cov)
    cov = np.dot(cov, m) / n
    return cov
