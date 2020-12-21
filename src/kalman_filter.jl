module Kalman
# implements the linear kalman filter algorithm
export KalmanFilter, predict, update, residual_of, measurement_of_state, batch_filter, rts_smoother

using LinearAlgebra

"""
    KalmanFilter(x_dim, z_dim; <keyword arguments>)

Defines a Kalman Filter of specified dimensions.
Use matrices instead of vectors for inputs.

...
#   Arguments
- `x_dim::Int`: number of dimensions of x
- `z_dim::Int`: number of dimensions of z

- `u_dim::Int = 0`: number of dimensions of u
- `x::AbstractArray = zeros(x_dim,1)`: state
- `P::AbstractArray = Matrix{Real}(I, x_dim, x_dim)`: uncertainty covariance
- `Q::AbstractArray = Matrix{Real}(I, x_dim, x_dim)`: process uncertainty
- `B::AbstractArray = Matrix{Real}(undef, x_dim, u_dim)`: control transition matrix
- `F::AbstractArray = Matrix{Real}(I, x_dim, x_dim)`: state transition matrix
- `H::AbstractArray = zeros(z_dim, x_dim)`: measurement function
- `R::AbstractArray = Matrix{Real}(I, z_dim, z_dim)`: state uncertainty
- `z::AbstractArray = Matrix{Real}(undef, z_dim, 1)`: measurement
- `K::AbstractArray = zeros(x_dim, z_dim)`: Kalman gain
- `y::AbstractArray = zeros(z_dim, 1)`: residual
- `S::AbstractArray = zeros(z_dim, z_dim)`: system uncertainty
- `SI::AbstractArray = zeros(z_dim, z_dim)`: inverse system uncertainty
- `x_prior::AbstractArray = copy(x)`: prior
- `P_prior::AbstractArray = copy(P)`: prior covariance
- `x_post::AbstractArray = copy(x)`: posterior
- `P_post::AbstractArray = copy(P)`: posterior covariance
...

# Examples
```julia-repl
julia> kf = KalmanFilter(x_dim=2, z_dim=1, x=[10., 4.5], P=[500. 0.; 0. 500.]);

julia> predict(kf, F=[1 0.3; 0 1], Q=0)

julia> println(kf.x, kf.P)\\
[11.35, 4.5][545.0 150.0; 150.0 500.0]

julia> predict(kf, F=[1 0.3; 0 1], Q=[0.588 1.175; 1.175 2.35])

julia> println(kf.x, kf.P)\\
[12.7, 4.5][680.588 301.175; 301.175 502.35]

julia> update(kf, [1.], R=5., H=[1 0])

julia> println(kf.x, kf.P)\\
[1.0853282146128578, -0.6397450072054944][4.963534951020146 2.1964722253014934;
2.1964722253014934 370.04549550896456]

"""
Base.@kwdef mutable struct KalmanFilter
    x_dim::Int
    z_dim::Int
    u_dim::Int = 0

    x::AbstractArray = zeros(x_dim,1)                   # state
    P::AbstractArray = Matrix{Real}(I, x_dim, x_dim)    # uncertainty covariance
    Q::AbstractArray = Matrix{Real}(I, x_dim, x_dim)    # process uncertainty
    B::AbstractArray = Matrix{Real}(undef, x_dim, x_dim)# control transition matrix
    F::AbstractArray = Matrix{Real}(I, x_dim, x_dim)    # state transition matrix
    H::AbstractArray = zeros(z_dim, x_dim)              # measurement function
    R::AbstractArray = Matrix{Real}(I, z_dim, z_dim)    # state uncertainty
    # M::AbstractArray = zeros(x_dim, z_dim)              # process-measurement cross correlation
    z::AbstractArray = Matrix{Real}(undef, z_dim, 1)    # measurement

    K::AbstractArray = zeros(x_dim, z_dim)              # Kalman gain
    y::AbstractArray = zeros(z_dim, 1)                  # residual
    S::AbstractArray = zeros(z_dim, z_dim)              # system uncertainty
    SI::AbstractArray = zeros(z_dim, z_dim)             # inverse system uncertainty

    x_prior::AbstractArray = copy(x)
    P_prior::AbstractArray = copy(P)
    x_post::AbstractArray = copy(x)
    P_post::AbstractArray = copy(P)
end

"""
    predict(filter; <keyword arguments>)

Predict next state (prior) using the Kalman filter state propagation equations

# Arguments
- `filter::KalmanFilter`: the filter

- `u = undef`: control vector
- `Q = nothing`: process uncertainty
- `B = nothing`: control transition matrix
- `F = nothing`: state transition matrix
- `modify::Bool = true`: whether to modify the filter or return x, P
"""
function predict(filter::KalmanFilter; u=undef, B=nothing, F=nothing, Q=nothing, modify::Bool=true)
    if (B == nothing) B = filter.B end
    if (F == nothing) F = filter.F end
    if (Q == nothing)
        Q = filter.Q
    elseif length(Q) == 1
        Q = I*Q
    end

    # propagate state x = Fx + Bu
    if isassigned(B) && isassigned(u)
        x = F*filter.x + B*u
    else
        x = F*filter.x
    end
    # propagate covariance P = FPF' + Q
    P = F*filter.P*F' + Q

    if modify
        # update state
        filter.x = copy(x)
        filter.P = copy(P)
        # update state prior
        filter.x_prior = copy(x)
        filter.P_prior = copy(P)
    else
        return x, P
    end
    return
end

"""
    update(filter, z; <keyword arguments>)

Update `filter` with new measurement `z`.

# Arguments
- `filter::KalmanFilter`: object of type `KalmanFilter`
- `z::AbstractArray`: measurement

- `H = nothing`: measurement function
- `R = nothing`: state uncertainty
- `modify::Bool = true`: whether to modify the filter or return x, P
"""
function update(filter::KalmanFilter, z::AbstractArray; R=nothing, H=nothing, modify=true)
    if (R == nothing)
        R = filter.R
    elseif length(R) == 1
        R = I*R
    end

    if (H == nothing)
        H = filter.H
    end

    # compute residual y = z - Hx
    y = z - H*filter.x
    # compute system uncertainty
    PHT = filter.P*H'
    S = H*PHT + R
    SI = inv(S)
    # compute Kalman gain K = PH'inv(S)
    K = PHT*SI
    # compute state x = x + Ky
    x = filter.x + K*y
    # compute state cov P = (I-KH)P(I-KH)' + KRK'
    I_KH = I - K*H
    P = I_KH*filter.P*I_KH' + K*R*K'

    if modify
        # update variables
        filter.x = copy(x)
        filter.P = copy(P)
        filter.y = copy(y)
        filter.K = copy(K)
        filter.S = copy(S)
        filter.SI = copy(SI)
        # save measurement and posterior state
        filter.z = copy(z)
        filter.x_post = copy(x)
        filter.P_post = copy(P)
    else
        return x, P
    end
    return
end

"""
    residual(filter, z)

Returns the residual for the given measurement `z`
without altering the state of the filter
"""
function residual_of(filter::KalmanFilter, z)
    return z - filter.H*filter.x_prior
end

"""
    measurement_of_state(filter, x)

Converts state `x` into measurement (does not alter the state of the filter)
"""
function measurement_of_state(filter::KalmanFilter, x)
    return self.H*x
end

"""
    batch_filter(filter, zs)

Updates `filter` with a sequence of measurements `zs`. Returns means and
covariances after prediction and update steps respectively
"""
function batch_filter(filter::KalmanFilter, zs; us=undef, Fs=nothing, Qs=nothing, Hs=nothing, Rs=nothing, Bs=nothing)
    n = size(zs)[1]

    if (Fs == nothing) Fs = fill(filter.F, n) end
    if (Qs == nothing) Qs = fill(filter.Q, n) end
    if (Hs == nothing) Hs = fill(filter.H, n) end
    if (Rs == nothing) Rs = fill(filter.R, n) end
    if (Bs == nothing) Bs = fill(filter.B, n) end
    if (us == undef) us = fill(undef, n) end

    means = fill(zeros(size(filter.x)), n)
    means_p = fill(zeros(size(filter.x)), n)

    covariances = fill(zeros(filter.x_dim, filter.x_dim), n)
    covariances_p = fill(zeros(filter.x_dim, filter.x_dim), n)

    for (i, z) in enumerate(eachrow(zs))
        predict(filter, u=us[i], B=Bs[i], F=Fs[i], Q=Qs[i])
        means_p[i] = filter.x
        covariances_p[i] = filter.P

        update(filter, z, R=Rs[i], H=Hs[i])
        means[i] = filter.x
        covariances[i] = filter.P
    end
    return means_p, covariances_p, means, covariances
end

"""
    rts_smoother(filter, Xs, Ps, Fs, Qs)

Runs the Rauch-Tung-Striebel Kalman smoother on a set of means and covariances
computed by a Kalman filter. Returns smoothed means `x`, smoothed state covariances
`P`, smoother gain `K` at each step and predicted state covariances `Pp`.
"""
function rts_smoother(filter::KalmanFilter, Xs::AbstractArray, Ps::AbstractArray; Fs=nothing, Qs=nothing)
    size(Xs)[1] == size(Ps)[1] || throw(DimensionMismatch("length of Xs and Ps must be the same"))

    n = size(Xs)[1]

    if (Fs == nothing) Fs = fill(filter.F, n) end
    if (Qs == nothing) Qs = fill(filter.Q, n) end

    # smoother gain
    K = fill(zeros(filter.x_dim, filter.x_dim),n)
    x, P, Pp = copy(Xs), copy(Ps), copy(Ps)

    for i in (n-1):-1:1
        Pp[i] = Fs[i+1]*P[i]*Fs[i+1]' + Qs[i+1]

        K[i] = P[i]*(Fs[i+1]')*inv(Pp[i])
        x[i] += K[i]*(x[i+1] - Fs[i+1]*x[i])
        P[i] += K[i]*(P[i+1] - Pp[i])*K[i]'
    end
    return (x, P, K, Pp)
end

"""
# Example

kf = KalmanFilter(x_dim=2, z_dim=1, R= [3.][:,:], Q=[0.02 0; 0 0.02], P=[500. 0.; 0. 49.], x=[0.; 0.][:,:], F=[1. 1.; 0. 1.], H=[1 0])

# simulate measurements
using Random
process_var = 0.02
sensor_var = 3.
dt = 1.
num = 50
xi = 0.0
vi = 1.0
process_std = sqrt(process_var)
sensor_std = sqrt(sensor_var)
rng = MersenneTwister(42)

xs = zeros(num) # track
zs = zeros(num) # measurements

for i in 1:num
    # move
    v = vi  + randn(rng)*process_std
    xi += v*dt
    xs[i] = xi
    # sense
    zs[i] = xi + randn(rng)*sensor_std
end

_, _, Xs, Covs = batch_filter(kf, zs)

Ms, Ps, _, _ = rts_smoother(kf, Xs, Covs)

using Plots
plotlyjs()
scatter(zs, label="measured")
plot!(collect(1:50),label="track")
plot!(hcat(Xs...)[1,:], label="filter")
plot!(hcat(Ms...)[1,:], label="smoother")
"""
end
