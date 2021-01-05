module UKF
# implements the unscented kalman filter and related functions
using LinearAlgebra
using StatsBase: mean, weights
using ._sigma_points
using Random
using .common

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
    Wm::AbstractWeights = weights(points_fn.Wm)
    Wc::AbstractWeights = weights(points_fn.Wc)

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

"""
############ example to check code ###########
function fx(x,dt)
    xout = zeros(size(x))
    xout[1] = x[2]*dt + x[1]
    xout[2] = x[2]
    return xout
end
function hx(x)
    return x[1:1]'
end

points_fn = JulierSigmaPoints(n=2, kappa=1)

ukf = UnscentedKF(x_dim=2, z_dim=1, dt=1.0, hx=hx, fx=fx, points_fn=points_fn)
ukf.x = [0. 0.]'
ukf.P *= 10.
ukf.R *= 0.5
ukf.Q = discrete_white_noise(2, dt=1., var=0.03)

num = 50
xs = zeros(num,2)
zs = zeros(num,1)
rng = MersenneTwister(42)

for i in 0:49
    z = Matrix{Real}(I,1,1)*(i + randn(rng)*0.5)
    predict(ukf)
    update(ukf,z)
    xs[i+1,:] = ukf.x
    zs[i+1,:] = ukf.z
end


scatter(zs, label="measured")
plot!(xs[:,1], label="filter")

######### example to check code ##########
points_fn = MerweScaledSigmaPoints(n=2, alpha=0.3, beta=2., kappa=0.1)
x = [0., 0.]
p = [32. 15.; 15. 40.]
sigmas = sigma_points(points_fn, x, p)

f_nonlinear_xy(x,y) = [x+y, 0.1*x^2 + y*y]

sigmas_f = zeros(5,2)
for i in 1:5
    sigmas_f[i,:] = f_nonlinear_xy(sigmas[i,1], sigmas[i,2])
end

ukf_mean, ukf_cov = unscented_transform(sigmas_f, points_fn.Wm, points_fn.Wc)

using BayesianFilters.stats
using Random

xy = rand(MvNormal(x, p), 5000)
fxy = f_nonlinear_xy.(xy[1,:], xy[2,:])
fxy = hcat(fxy...)
scatter(fxy[1,:],fxy[2,:], label="transformed distribution", markersize=0.5)
scatter!([mean(fxy[1,:])], [mean(fxy[2,:])], label="computed mean", markersize=5)
scatter!([ukf_mean[1]], [ukf_mean[2]], label="unscented mean", markersize=2)
ukf_mean[1] - mean(fxy[1,:])
ukf_mean[2] - mean(fxy[2,:])
"""
end
