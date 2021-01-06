module UKF
# implements the unscented kalman filter and related functions
include("common.jl")
using .common
include("_sigma_points.jl")
using ._sigma_points

export UnscentedKF, predict, update, batch_filter, rts_smoother
export unscented_transform, compute_process_sigmas, cross_variance
export MerweScaledSigmaPoints, JulierSigmaPoints, sigma_points

using LinearAlgebra
using StatsBase: mean, weights


"""
    unscented_transform(sigmas, Wm, Wc, noise_cov=nothing, mean_fn=nothing)

Computes unscented transform of a set of sigma points and weights
and returns the mean and covariance.

"""
function unscented_transform(sigmas, Wm, Wc; noise_cov=nothing, mean_fn::Function=mean, residual_fn::Function=-)
    kmax, n = size(sigmas)

    x = mean_fn(sigmas, weights(Wm), dims=1)

    if residual_fn == -
        y = sigmas .- x
        P = (y')*Diagonal(Wc)*y
    else
        P = zeros(n,n)
        for k in 1:kmax
            y = residual_fn.(sigmas[k,:], x)
            P += Wc[k] * (y*y')
        end
    end

    if noise_cov != nothing
        P += noise_cov
    end
    return x[:], P
end

Base.@kwdef mutable struct UnscentedKF
    x_dim::Int
    z_dim::Int
    dt::Real
    fx::Function
    hx::Function
    points_fn::Union{MerweScaledSigmaPoints,JulierSigmaPoints}
    sqrt_fn::Function = cholesky
    x_mean_fn::Function = mean
    z_mean_fn::Function = mean
    residual_x::Function = -
    residual_z::Function = -
    state_add::Function = +

    x::AbstractArray = zeros(x_dim,1)
    P::AbstractArray = Matrix{Real}(I, x_dim, x_dim)
    Q::AbstractArray = Matrix{Real}(I, x_dim, x_dim)
    R::AbstractArray = Matrix{Real}(I, z_dim, z_dim)
    num_sigmas::Real = points_fn.num_sigmas
    Wm::AbstractArray = points_fn.Wm
    Wc::AbstractArray = points_fn.Wc

    sigmas_f::AbstractArray = zeros(num_sigmas, x_dim)
    sigmas_h::AbstractArray = zeros(num_sigmas, z_dim)

    K::AbstractArray = zeros(x_dim, z_dim)
    y::AbstractArray = zeros(z_dim, 1)
    z::AbstractArray = Matrix(undef, z_dim, 1)
    S::AbstractArray = zeros(z_dim, z_dim)
    SI::AbstractArray = zeros(z_dim, z_dim)

    x_prior::AbstractArray = copy(x)
    P_prior::AbstractArray = copy(P)
    x_post::AbstractArray = copy(x)
    P_post::AbstractArray = copy(P)
end

function compute_process_sigmas(filter::UnscentedKF, dt; fx=filter.fx)
    sigmas = sigma_points(filter.points_fn, filter.x, filter.P)

    for (i,s) in enumerate(eachrow(sigmas))
        filter.sigmas_f[i,:] = fx(s, dt)
    end
end

function cross_variance(filter, x, z, sigmas_f, sigmas_h)
    Pxz = zeros(size(sigmas_f)[2], size(sigmas_h)[2])
    N = size(sigmas_f)[1]
    for i in 1:N
        dx = filter.residual_x(sigmas_f[i,:], x)
        dz = filter.residual_z(sigmas_h[i,:], z)
        Pxz += filter.Wc[i]*(dx)*(dz')
    end
    return Pxz
end

function predict(filter::UnscentedKF;
    UT::Function=unscented_transform,
    dt::Real=filter.dt,
    fx::Function=filter.fx, hx::Function=filter.hx)

    compute_process_sigmas(filter, dt, fx=fx)
    filter.x, filter.P = UT(filter.sigmas_f, filter.Wm, filter.Wc, noise_cov=filter.Q, mean_fn=filter.x_mean_fn, residual_fn=filter.residual_x)

    # same output as filterpy if the following line is commented
    # likely a bug in filterpy
    filter.sigmas_f = sigma_points(filter.points_fn, filter.x, filter.P)

    filter.x_prior = copy(filter.x)
    filter.P_prior = copy(filter.P)
end

function update(filter::UnscentedKF, z; R=filter.R, UT=unscented_transform, hx=filter.hx)
    if length(R) == 1
        R = I*R
    end

    sigmas_h = zeros(size(filter.sigmas_f)[1], filter.z_dim)
    for (i,s) in enumerate(eachrow(filter.sigmas_f))
        sigmas_h[i,:] = hx(s)
    end

    filter.sigmas_h = copy(sigmas_h)

    zp, filter.S = UT(filter.sigmas_h, filter.Wm, filter.Wc, noise_cov=R, mean_fn=filter.z_mean_fn, residual_fn=filter.residual_z)
    filter.SI = inv(filter.S)

    Pxz = cross_variance(filter, filter.x, zp, filter.sigmas_f, filter.sigmas_h)

    filter.K = Pxz*filter.SI
    filter.y = filter.residual_z(z,zp)

    filter.x = filter.state_add(filter.x, filter.K*filter.y)
    filter.P = filter.P - filter.K*filter.S*filter.K'

    filter.z = copy(z)
    filter.x_post = copy(filter.x)
    filter.P_post = copy(filter.P)
end

function batch_filter(filter::UnscentedKF, zs; Rs=fill(filter.R, size(zs)[1]),
    dts=fill(filter.dt, size(zs)[1]), UT=unscented_transform)

    size(zs)[2] == filter.z_dim || throw(DimensionMismatch("size(zs)[2] must be equal to filter.z_dim"))
    size(zs)[1] == size(Rs)[1] || throw(DimensionMismatch("Rs must be an array of arrays with size(Rs)[1] == size(zs)[1]"))

    n = size(zs)[1]

    means = zeros(n, filter.x_dim)
    covariances = fill(zeros(filter.x_dim, filter.x_dim),n)

    for (i,z) in enumerate(eachrow(zs))
        predict(filter, dt=dts[i], UT=UT)
        update(filter, z, R=Rs[i], UT=UT)
        means[i,:] = filter.x
        covariances[i] = filter.P
    end
    return (means, covariances)
end

function rts_smoother(filter::UnscentedKF, Xs, Ps; Qs=fill(filter.Q, size(Xs)[1]),
    dts=fill(filter.dt, size(Xs)[1]), UT=unscented_transform)

    size(Xs)[1] == size(Ps)[1] || throw(DimensionMismatch("length of Xs and Ps must be the same"))
    n, x_dim = size(Xs)

    Ks = fill(zeros(x_dim, x_dim),n)
    num_sigmas = filter.num_sigmas
    xs, ps = copy(Xs), copy(Ps)
    sigmas_f = zeros(num_sigmas, x_dim)

    for k in (n-1):-1:1
        sigmas = sigma_points(filter.points_fn, xs[k,:], ps[k])

        for i in 1:num_sigmas
            sigmas_f[i,:] = filter.fx(sigmas[i,:], dts[k])
        end

        xb, Pb = UT(sigmas_f, filter.Wm, filter.Wc, noise_cov=filter.Q, mean_fn=filter.x_mean_fn, residual_fn=filter.residual_x)

        Pxb = zeros(size(sigmas_f)[2], size(sigmas_f)[2])
        for i in 1:num_sigmas
            y = filter.residual_x(sigmas_f[i,:], xb)
            z = filter.residual_x(sigmas[i,:], Xs[k,:])
            Pxb += filter.Wc[i]*(z)*(y')
        end
        K = Pxb*inv(Pb)

        xs[k,:] += K*filter.residual_x(xs[k+1,:], xb)
        ps[k] += (K*(ps[k+1] .- Pb))*K'
        Ks[k] = K
    end
    return (xs, ps, Ks)
end

end
