module common
# contains some common useful functions
export discrete_white_noise, continuous_white_noise

"""
    discrete_white_noise(dim, dt=1., var=1.)

Returns the Q matrix for Discrete Constant White Noise Model. `dim` ∈ (2,3,4) is
the number of dimensions, dim = 2 for constant velocity, dim = 3 for constant
acceleration and dim = 4 for constant jerk, `dt` is the time step and `var` is
the variance in the noise.

# Example
```julia-repl
julia> discrete_white_noise(2, dt=0.1, var=1.)
2×2 Array{Float64,2}:
 2.5e-5  0.0005
 0.0005  0.01
"""
function discrete_white_noise(dim::Int; dt::Real=1., var::Real=1.)
    dim in (1, 2, 3) || throw(DimensionMismatch("dim must be between 2 and 4"))

    if dim==2
        Q = [.25*dt^4 .5*dt^3;
             .5*dt^3 dt^2]
    elseif dim==3
        Q = [.25*dt^4 .5*dt^3 .5*dt^2;
             .5*dt^3 dt^2 dt;
             .5*dt^2 dt 1.]
    else
        Q = [(dt^6)/36 (dt^5)/12 (dt^4)/6 (dt^3)/6;
             (dt^5)/12 (dt^4)/4 (dt^3)/2 (dt^2)/2;
             (dt^4)/6 (dt^3)/2 dt^2 dt;
             (dt^3)/6 (dt^2)/2 dt 1.]
    end
    return Q.*var
end

"""
    continuous_white_noise(dim, dt=1., S=1.)

Returns the Q matrix for Discretized Continuous White Noise Model. `dim` ∈ (2,3,4)
is the number of dimensions, dim = 2 for constant velocity, dim = 3 for constant
acceleration and dim = 4 for constant jerk, `dt` is the time step and `S` is
the spectral density for the continuous process.

# Examples
```julia-repl
julia> continuous_white_noise(2, dt=0.1, S=1.)
2×2 Array{Float64,2}:
 0.000333333  0.005
 0.005        0.1
"""
function continuous_white_noise(dim::Int; dt::Real=1., S::Real=1.)
    dim in (1, 2, 3) || throw(DimensionMismatch("dim must be between 2 and 4"))

    if dim==2
        Q = [(dt^3)/3 (dt^2)/2;
             (dt^2)/2 dt]
    elseif dim==3
        Q = [(dt^5)/20. (dt^4)/8. (dt^3)/6.;
             (dt^4)/8. (dt^3)/3. (dt^2)/2.;
             (dt^3)/6. (dt^2)/2. dt]
    else
        Q = [(dt^7)/252. (dt^6)/72. (dt^5)/30. (dt^4)/24.;
             (dt^6)/72.  (dt^5)/20. (dt^4)/8.  (dt^3)/6.;
             (dt^5)/30.  (dt^4)/8.  (dt^3)/3.  (dt^2)/2.;
             (dt^4)/24.  (dt^3)/6.  (dt^2)/2.   dt]
    end
    return Q.*S
end

end
