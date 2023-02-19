# Optimization

## Mean-Variance Optimization

### Analytical Solutions

This problem is a linear optimization problem with equality constraints. We have:

$$
\min_{x_{1}, x_{2}, ... x_{n}} f(x_{1}, x_{2}, ..., x_{n})\\
$$

subject to:

$$
g_{1}(x_{1}, x_{2}, ..., x_{n}) = b_{1}\\
$$

```math
\vdots
```

$$
g_{n}(x_{1}, x_{2}, ..., x_{n}) = b_{n}\\
$$

The Langrangian function would be:

```math
L(x, \lambda) = f(\textbf{x}) + \sum_{j=1}^{m} \lambda_{j}g_{j}(x) - b_{j}
```

We then minimize the Lagrangian:

```math
\min_{x, \lambda} L(x, \lambda)
```

### Classical mean-variance problem with a target rate of return

In the classical mean-variance problem with a target rate of return, the optimization function is the variance subject to constraints on the targeted rate of returns and the sum of weights.

```math
\min_{w} \frac{1}{2}w^{T}\Sigma w
```

subject to:

$$
w^{T}\mu = m
$$

$$
w^{T} 1 = 1
$$

This problem is an optimization with equality constraints. We can solve it using the method of Lagrange. We form the Lagrange function with two Lagrange multipliers $\lambda$ and $\gamma$:

$$
L(w, \lambda, \gamma) = \frac{1}{2}w^{T}\Sigma w + \lambda(m - w^{T}\mu) + \gamma(1 - w^{T}*1)
$$

$$
\frac{\partial L}{\partial w}(w, \lambda, \gamma) = \Sigma w - \lambda\mu -\gamma 1 = 0
$$
$$
\hat{w} = \Sigma^{-1}(\lambda\mu + \gamma * 1)
$$





