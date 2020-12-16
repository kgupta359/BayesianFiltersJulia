module ghfilter

export GHFilter, GHKFilter, update, batch_filter

FloatOrNothing = Union{Real, Nothing}
FloatOrArray = Union{Real, AbstractArray}

"""
    GHFilter(x, dx, dt, g, h)

Define and initialize a g-h filter

# Examples
```julia-repl
julia> ghf = GHFilter(0.0, 0.0, 1.0, 0.8, 0.2)
GHFilter(0.0, 0.0, 1.0, 0.8, 0.2)
```
"""
mutable struct GHFilter
    x::FloatOrArray
    dx::FloatOrArray
    dt::Real
    g::Real
    h::Real
end


"""
    update(
            filter::GHFilter,
            z::Union{Real, AbstractArray{Real}},
            g::Union{Real, Nothing}=nothing,
            h::Union{Real, Nothing}=nothing
            )

performs the g-h filter predict and update step on the
measurement z and returns the state of x and dx as a tuple.

Optional: Overrides filter.g and filter.h for this update.

# Examples
```julia-repl
julia> ghf = GHFilter(0.0, 0.0, 1.0, 0.8, 0.2);

julia> update(ghf, 1.2)
(0.96, 0.24)

julia> update(ghf, 2.1, g=0.85, h=0.15)
(1.965, 0.375)
```
"""
function update(filter::GHFilter, z::FloatOrArray; g::FloatOrNothing=nothing, h::FloatOrNothing=nothing)
    if g == nothing
        g = filter.g
    end
    if h == nothing
        h = filter.h
    end

    # prediction step
    dx_prediction = filter.dx
    x_prediction = filter.x .+ filter.dx*filter.dt
    # update step
    y = z .- x_prediction
    filter.dx = dx_prediction .+ h* y/ filter.dt
    filter.x = x_prediction .+ g*y
    return filter.x, filter.dx
end

"""
    batch_filter(filter::GHFilter, data::AbstractArray, save_predictions::Bool=false)

Given a sequenced list of data, performs g-h filter with a fixed g and h.

# Examples
```julia-repl
julia> ghf = GHFilter(0.0, 0.0, 1.0, 0.8, 0.2)
GHFilter(0.0, 0.0, 1.0, 0.8, 0.2)

julia> batch_filter(ghf, [1.0, 2.0, 3.0])
([0.0; 0.8; 1.8; 2.84], [0.0; 0.2; 0.4; 0.56])

julia> ghf = GHFilter([1., 0.], [2., 3.], 1., 0.1, 0.02);

julia> batch_filter(ghf, [1.0 0.2; 2.0 3.0; 3.0 1.0])
([1.0 0.0; 2.8 2.72; 4.484 5.3976; 6.04992 7.559488],
 [2.0 3.0; 1.96 2.944; 1.9048 2.89072; 1.837024 2.7449536])
```
"""
function batch_filter(filter::GHFilter, data::AbstractArray; save_predictions::Bool=false)
    if length(filter.x) == 1
        x = [filter.x]
        dx = [filter.dx]
    else
        x = filter.x
        dx = filter.dx
    end
    dim = length(x)
    n = size(data)[1]

    x_results = zeros(n+1, dim)
    dx_results = zeros(n+1, dim)

    x_results[1,:] = x
    dx_results[1,:] = dx

    if save_predictions
        predictions = zeros(n,dim)
    end

    h_dt = filter.h/filter.dt

    for (i, z) in enumerate(eachrow(data))
        # prediction
        x_est = x .+ dx*filter.dt

        # update
        residual = z .- x_est
        dx = dx .+ h_dt*residual
        x = x_est .+ filter.g*residual

        x_results[i+1,:] = x
        dx_results[i+1,:] = dx

        if save_predictions
            predictions[i,:] = x_est
        end
    end

    if save_predictions
        return x_results, dx_results, predictions
    end
    return x_results, dx_results
end

"""
    GHKFilter(x, dx, ddx, dt, g, h, k)

Define and initialize a g-h-k filter

# Examples
```julia-repl
julia> x0 = [1.0, 10.0, 100.0];

julia> dx0 = [10.0, 12.0, 0.2];

julia> ddx0 = [0.1, 0.2, 0.0];

julia> filter = GHKFilter(x0, dx0, ddx0, 1.0, 0.8, 0.2, 0.1)
GHKFilter([1.0, 10.0, 100.0], [10.0, 12.0, 0.2], [0.1, 0.2, 0.0], 1.0, 0.8, 0.2, 0.1)
```
"""
mutable struct GHKFilter
    x::FloatOrArray
    dx::FloatOrArray
    ddx::FloatOrArray
    dt::Real
    g::Real
    h::Real
    k::Real
end

"""
    update(
            filter::GHKFilter,
            z::Union{Real, AbstractArray},
            g::Union{Real, Nothing}=nothing,
            h::Union{Real, Nothing}=nothing,
            k::Union{Real, Nothing}=nothing,
            )

performs the g-h-k filter predict and update step on the
measurement z and returns the state of x and dx as a tuple.
Optional: Overrides filter.g, filter.h and filter.k for this update.

# Examples
```julia-repl
julia> filter = GHKFilter(x0, dx0, ddx0, 1.0, 0.8, 0.2, 0.1)
GHKFilter([1.0, 10.0, 100.0], [10.0, 12.0, 0.2], [0.1, 0.2, 0.0], 1.0, 0.8, 0.2, 0.1)

julia> update(filter, [12.0, 11.3, 105.9])
([11.81, 13.46, 104.76], [10.29, 10.04, 1.34])
```
"""
function update(filter::GHKFilter, z::FloatOrArray; g::FloatOrNothing=nothing, h::FloatOrNothing=nothing, k::FloatOrNothing=nothing)
    if g == nothing
        g = filter.g
    end
    if h == nothing
        h = filter.h
    end
    if k == nothing
        k = filter.k
    end

    dt = filter.dt
    dt_sqr = dt^2

    # prediction
    ddx_prediction = filter.ddx
    dx_prediction = filter.dx .+ filter.ddx*dt
    x_prediction = filter.x .+ filter.dx*dt .+ 0.5*filter.ddx*dt_sqr

    # update
    y = z .- x_prediction
    filter.ddx = ddx_prediction + 2*k*y/dt_sqr
    filter.dx = dx_prediction + h*y/dt
    filter.x = x_prediction + g*y

    return filter.x, filter.dx
end

"""
    batch_filter(filter::GHKFilter, data::AbstractArray, save_predictions::Bool=false)

Given a sequenced list of data, performs g-h-k filter with fixed g, h and k.

# Examples
```julia-repl
julia> filter = GHKFilter(x0, dx0, ddx0, 1.0, 0.8, 0.2, 0.1);

julia> z = [12.0 11.3 105.9; 14.0 15.3 111.9];

julia> batch_filter(filter, z)
([1.0 10.0 100.0; 11.81 13.46 104.76; 15.649 16.744 110.854],
 [10.0 12.0 0.2; 10.29 10.04 1.34; 8.931 6.636 3.526],
 [0.1 0.2 0.0; 0.29 -1.96 1.14; -1.359 -3.404 2.186])
```
"""
function batch_filter(filter::GHKFilter, data::AbstractArray; save_predictions::Bool=false)
    if length(filter.x) == 1
        x = [filter.x]
        dx = [filter.dx]
        ddx = [filter.ddx]
    else
        x = filter.x
        dx = filter.dx
        ddx = filter.ddx
    end

    dim = length(x)
    n = size(data)[1]

    x_results = zeros(n+1, dim)
    dx_results = zeros(n+1, dim)
    ddx_results = zeros(n+1, dim)
    x_results[1,:] = x
    dx_results[1,:] = dx
    ddx_results[1,:] = ddx

    if save_predictions
        predictions = zeros(n,dim)
    end

    dt = filter.dt
    dt_sqr = dt^2
    h_dt = filter.h/filter.dt
    k_dt_sqr = 2*filter.k/(dt^2)

    for (i, z) in enumerate(eachrow(data))
        # prediction
        dx_est = dx .+ ddx*dt
        x_est = x .+ dx*filter.dt .+ 0.5*ddx*dt_sqr

        # update
        y = z .- x_est
        ddx = ddx .+ k_dt_sqr*y
        dx = dx_est .+ h_dt*y
        x = x_est .+ filter.g*y

        x_results[i+1,:] = x
        dx_results[i+1,:] = dx
        ddx_results[i+1,:] = ddx

        if save_predictions
            predictions[i,:] = x_est
        end
    end

    if save_predictions
        return x_results, dx_results, ddx_results, predictions
    end

    return x_results, dx_results, ddx_results
end

end
