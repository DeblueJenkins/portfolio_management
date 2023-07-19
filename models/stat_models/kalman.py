""" Consider the folowing system:
Model:
x(k+1) = A * x(k) + b + w(k)
Measurement:
y(k) = C * x(k) + d + eps(k)
where:
m: # of state variables
A: m x m
b: m x 1
x: m x 1 state vector
Q: m x m cov matrix of w(k)
n: # of observed variables
C: n x m
d: n x 1
y: n x 1 observation vector
R: n x n cov matrix of eps(k)
INPUT:
A, b, C, d, Q, R as in the model above
x0: state vector initial value
s0: sigma matrix initial value
y: observed data
OUTPUT:
xPred: prediction vector
xUpdate: updated state vector
SigmaPred: cov Matrix of the prediction
SigmaUpdate: cov Matrix of the update
e: prediction error
Sigmay: cov Matrix of the prediction error
The Kalman Filter is a recursive data processing algorithm which gets the
system measured data (y(t) above) and recursively estimates the "real value"
"""
from numpy import zeros
from numpy import shape
from numpy import dot
from numpy import eye
from numpy.linalg import inv


def linear_kalman(y, A, b, C, d, Q, R, x0, s0):
    N = len(y[0, :]) # number of observations
    [n, m] = shape(C)
    xPred = zeros((m, N))
    xUpdate = zeros((m, N))
    SigmaPred = zeros((m, m, N))
    SigmaUpdate = zeros((m, m, N))
    SigmaUpdate[:, :, 0] = s0
    e = zeros((n, N))
    Sigmay = zeros((n, n, N))
    xUpdate[:, 0] = x0
    I = eye(m)
    for i in range(1, N):
        # Prediction Step
        xPred[:, i] = dot(A, xUpdate[:, i-1]).T + b.T
        SigmaPred[:, :, i] = dot(A, dot(SigmaUpdate[:, :, i-1], A.T)) + Q
        Sigmay[:, :, i] = dot(C, dot(SigmaPred[:, :, i], C.T)) + R
        # Update Step
        K = dot(SigmaPred[:, :, i], dot(C.T, inv(Sigmay[:, :, i])))
        e[:, i] = y[:, i] - dot(C, xPred[:, i]) - d
        xUpdate[:, i] = xPred[:, i] + dot(K, e[:, i])
        SigmaUpdate[:, :, i] = dot((I - dot(K, C)), SigmaPred[:, :, i])
    return xPred, xUpdate, SigmaPred, SigmaUpdate, e, Sigmay