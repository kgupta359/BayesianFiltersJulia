module _sigma_points
# generate sigma points for unscented transform
export MerweScaledSigmaPoints, JulierSigmaPoints, sigma_points

using LinearAlgebra

Base.@kwdef struct MerweScaledSigmaPoints
    n::Int
    alpha::Real
    beta::Real
    kappa::Real = 0.0
    sqrt_fn::Function = cholesky
    subtract::Function = -
    num_sigmas::Int = 2n+1
    Wm::AbstractVector = zeros(num_sigmas)
    Wc::AbstractVector = zeros(num_sigmas)

    function MerweScaledSigmaPoints(n, alpha, beta, kappa, sqrt_fn, subtract, num_sigmas, Wm, Wc)
        # compute weights
        lambda = (alpha^2) * (n + kappa) - n
        c = 0.5/(n+lambda)
        Wc[1] = lambda/(n + lambda) + (1 - alpha^2 + beta)
        Wc[2:end] .= c
        Wm[1] = lambda/(n + lambda)
        Wm[2:end] .= c
        new(n, alpha, beta, kappa, sqrt_fn, subtract, num_sigmas, Wm, Wc)
    end
end

function sigma_points(method::MerweScaledSigmaPoints, x::AbstractArray, P::AbstractArray)
    method.n == length(x) || throw(DimensionMismatch("expected size(x) = $(method.n), got $(length(x))"))
    if length(P) == 1
        P = I*P
    end
    n = method.n
    lambda = (method.alpha^2) * (n + method.kappa) - n
    U = method.sqrt_fn(Symmetric((lambda + n)*P)).U
    sigmas = zeros(2n+1, n)
    sigmas[1,:] = x
    for k in 1:n
        sigmas[k+1, :] = method.subtract(x[:], -U[k,:])
        sigmas[n+k+1, :] = method.subtract(x[:], U[k,:])
    end
    return sigmas
end

Base.@kwdef struct JulierSigmaPoints
    n::Int
    kappa::Real = 0.0
    sqrt_fn::Function = cholesky
    subtract::Function = -
    num_sigmas::Int = 2n+1
    Wm::AbstractVector = zeros(num_sigmas)
    Wc::AbstractVector = zeros(num_sigmas)

    function JulierSigmaPoints(n, kappa, sqrt_fn, subtract, num_sigmas, Wm, Wc)
        Wm[1] = kappa/(n+kappa)
        Wm[2:end] .= 0.5/(n+kappa)
        Wc = copy(Wm)
        new(n, kappa, sqrt_fn, subtract, num_sigmas, Wm, Wc)
    end
end

function sigma_points(method::JulierSigmaPoints, x::AbstractArray, P::AbstractArray)
    method.n == length(x) || throw(DimensionMismatch("expected size(x) = $(method.n), got $(length(x))"))
    if length(P) == 1
        P = I*P
    end
    n = method.n
    U = method.sqrt_fn(Symmetric((method.kappa + n)*P)).U
    sigmas = zeros(2n+1, n)
    sigmas[1,:] = x
    for k in 1:n
        sigmas[k+1, :] = method.subtract(x[:], -U[k,:])
        sigmas[n+k+1, :] = method.subtract(x[:], U[k,:])
    end
    return sigmas
end

end
