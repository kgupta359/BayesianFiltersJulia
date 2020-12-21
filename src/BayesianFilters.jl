module BayesianFilters

include("ghfilter.jl")
using .ghfilter

include("discrete_bayes.jl")
using .discrete_bayes

include("stats.jl")
using .stats

include("kalman_filter.jl")
using .Kalman

include("common.jl")
using .common
end
