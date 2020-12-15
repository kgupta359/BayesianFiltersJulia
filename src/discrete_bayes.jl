module discrete_bayes

using DSP

export normalize, update, predict

"""
    normalize(pdf::Array{Float64})

Returns normalized discrete probability distribution function

# Examples
```julia-repl
julia> belief = [0.3 0.1 0.1 0.1 0.0 0.1 0.4 0.4 0.1 0.0];

julia> normalize(belief)
1×10 Array{Float64,2}:
 0.1875  0.0625  0.0625  0.0625  0.0  0.0625  0.25  0.25  0.0625  0.0
```
"""
function normalize(pdf::Array{Float64})
    pdf /= sum(pdf)
    return pdf
end

"""
    update(likelihood::Array{Float64}, prior::Array{Float64})

Computes the posterior of a discrete random variable
given a discrete likelihood and prior

# Examples
```julia-repl
julia> belief = [0.3 0.1 0.1 0.1 0.0 0.1 0.4 0.4 0.1 0.0];

julia> likelihood = [1. 0. 1. 0.3 0.3 0.1 0.001 0.9 0.1 0.9];

julia> update(likelihood, belief)
1×10 Array{Float64,2}:
 0.370188  0.0  0.123396  0.0370188  0.0  0.0123396  0.000493583  0.444225  0.0123396  0.0
```
"""
function update(likelihood::Array{Float64}, prior::Array{Float64})
    posterior = prior .* likelihood
    return normalize(posterior)
end

"""
    predict(pdf::Array{Float64}, offset::Int64, kernel::Array{Float64})

Performs the discrete Bayes filter prediction step, generating the prior.

`pdf` is a discrete probability distribution expressing our initial belief.

`offset` is an integer specifying how much we want to move to the right
(negative values means move to the left)

Noise in that offset is expressed in `kernel`.

# Examples
```julia-repl
julia> belief = [.05 .05 .05 .05 .55 .05 .05 .05 .05 .05];

julia> kernel = [.1 .8 .1];

julia> prior = predict(belief, 1, kernel)
10-element Array{Float64,1}:
 0.05  0.05  0.05  0.05  0.1  0.45  0.1  0.05  0.05  0.05
```
"""
function predict(pdf::Array{Float64}, offset::Int64, kernel::Array{Float64})
    n = length(pdf)
    m = length(kernel)
    width = Int((m-1)/2)
    prior = zeros(n)
    for i in 1:n
        for j in 1:m
            index = (i + (width-j) - offset + n) % n
            prior[i] += pdf[index+1] .* kernel[j]
        end
    end
    return prior
end

end
