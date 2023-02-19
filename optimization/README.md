# Optimization

## Mean-Variance Optimization

### Analytical Solutions

### Classical mean-variance problem with a target rate of return

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
\min_{\textbf{x}, \textbf{\lambda}} L(\textbf{x}, \textbf{\lambda})
```
