module stats
# implements some common statistical functions
export mahalanobis, log_pdf, log_likelihood, likelihood,
export gaussian, mul, add, covariance_ellipse
export Normal, MvNormal, pdf, mean, var, cov

using Statistics
using Distributions
using LinearAlgebra


FloatOrArray = Union{Real, AbstractArray}

"""
    mahalanobis(x, y, cov)

Returns the mahalanobis distance between `x` and `y`

# Examples
```julia-repl
julia> mahalanobis(3., 3.5, 4.0^2) # univariate case
0.125

julia> mahalanobis(3., 6., 1.) # univariate, 3 std away
3.0

julia> mahalanobis([1., 2], [1.1, 3.5], [1. .1; .1 13])
0.4253332705891392
```
"""
function mahalanobis(x::FloatOrArray, y::FloatOrArray, cov::FloatOrArray)
    size(x) == size(y) || throw(DimensionMismatch("size of x not equal to size of y"))
    dist = sqrt((y .- x)'*inv(cov)*(y .- x))
    return dist
end

"""
    log_pdf(d::Distribution, x::Union{Real, AbstractArray{Real}})

Computes the log of the probability density function of the normal distribution
N(mean, cov) for the data x. The normal may be univariate or multivariate.
"""
log_pdf(d::Distribution, x::FloatOrArray) = Distributions.logpdf(d, x)

"""
    log_likelihood(z, x, P, H, R)

Returns log-likelihood of the measurement z given the Gaussian
posterior (x, P) using measurement function H and measurement
covariance error R
"""
log_likelihood(z::Real, x::Real, P::Real, H::Real, R::Real) = log_pdf(Normal(H*x, sqrt(H*P*H + R)), z)
log_likelihood(z::AbstractArray, x::AbstractArray, P::AbstractArray, H::AbstractArray, R::AbstractArray) = log_pdf(MvNormal(H*x, H*P*H' + R), z)

"""
    likelihood(z, x, P, H, R)

Returns likelihood of the measurement z given the Gaussian
posterior (x, P) using measurement function H and measurement
covariance error R
"""
likelihood(z, x, P, H, R) = exp(log_likelihood(z, x, P, H, R))

"""
    gaussian(x, mean, var)

Returns probability density function (pdf) for x given a Gaussian with the
specified mean and variance.
"""
gaussian(x::AbstractArray, mean::Real, var::Real) = pdf.(Normal(mean, sqrt(var)), x)
gaussian(x::AbstractArray, mean::AbstractArray, cov::AbstractArray) = pdf.(MvNormal(mean, cov), x)

"""
    mul(d1:: Distribution, d2:: Distribution)

Returns the product of the two gaussian distributions
"""
function mul(d1::UnivariateDistribution, d2::UnivariateDistribution)
    mean1 = Statistics.mean(d1)
    mean2 = Statistics.mean(d2)

    size(mean1) == size(mean2) || throw(DimensionMismatch("size of distributions must be equal"))

    var1 = Statistics.var(d1)
    var2 = Statistics.var(d2)

    var3 = 1.0/(1.0/var1 + 1.0/var2)
    mean3 = (var1*mean2 + var2*mean1)/(var1 + var2)

    return Normal(mean3, sqrt(var3))
end

function mul(d1::MultivariateDistribution, d2::MultivariateDistribution)
    mean1 = Statistics.mean(d1)
    mean2 = Statistics.mean(d2)

    size(mean1) == size(mean2) || throw(DimensionMismatch("size of distributions must be equal"))

    var1 = Statistics.cov(d1)
    var2 = Statistics.cov(d2)

    sum_inv = inv(var1 + var2)
    var3 = var1*sum_inv*var2
    mean3 = var2*sum_inv*mean1 + var1*sum_inv*mean2

    return MvNormal(mean3, var3)
end

"""
    add(d1::Distribution, d2::Distribution)

Returns the sum of the two gaussian distributions
"""
function add(d1::UnivariateDistribution, d2::UnivariateDistribution)
    mean1 = Statistics.mean(d1)
    mean2 = Statistics.mean(d2)
    size(mean1) == size(mean2) || throw(DimensionMismatch("size of distributions must be equal"))
    var1 = Statistics.var(d1)
    var2 = Statistics.var(d2)

    mean3 = mean1 + mean2
    var3 = var1 + var2

    return Normal(mean3, sqrt(var3))
end

function add(d1::MultivariateDistribution, d2::MultivariateDistribution)
    mean1 = Statistics.mean(d1)
    mean2 = Statistics.mean(d2)
    size(mean1) == size(mean2) || throw(DimensionMismatch("size of distributions must be equal"))
    var1 = Statistics.cov(d1)
    var2 = Statistics.cov(d2)

    mean3 = mean1 .+ mean2
    var3 = var1 .+ var2

    return MvNormal(mean3, var3)
end

"""
    covariance_ellipse(P, deviations=1.0)

Returns a tuple (angle, width, height) defining the covariance ellipse
representing the 2 dimensional covariance matrix P
"""
function covariance_ellipse(P::AbstractArray; deviations::Real=1.0)
    size(P) == (2, 2) || throw(DimensionMismatch("P must be a 2x2 matrix"))
    U, s, _ = svd(P)
    orientation = atan(U[2,1], U[1,1])
    width = deviations*sqrt(s[1])
    height = deviations*sqrt(s[2])

    height < width || throw(ErrorException("width must be greater than height"))

    return (orientation, width, height)
end

end
