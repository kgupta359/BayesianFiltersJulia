module BayesianFilters

using Reexport

include("ghfilter.jl")
@reexport using ghfilter

include("discrete_bayes.jl")
@reexport using discrete_bayes

end
