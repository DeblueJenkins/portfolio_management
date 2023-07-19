import numpy as np

corr = np.array([[ 1.   , 0.174, 0.091,  0.07 ],
                 [0.174,  1.   ,  0.914,  0.846],
                 [0.091,  0.914,  1.   ,  0.911],
                 [ 0.07 ,  0.846,  0.911,  1.   ]])

mean = np.array([-0.66591685, -0.69220712, -0.72482504, -0.71915904])

X = np.random.multivariate_normal(mean=mean, cov=corr, size=100)