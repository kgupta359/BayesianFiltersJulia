module BayesianFilters

include("ghfilter.jl")
using .ghfilter

include("discrete_bayes.jl")
using .discrete_bayes

include("stats.jl")
using .stats

include("kalman.jl")
using .kalman

include("common.jl")
using .common

include("UKF.jl")
using .UKF

end
