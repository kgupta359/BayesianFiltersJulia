module discrete_bayes

using DSP

export normalize, update, predict

function normalize(pdf::Array{Float64})
    pdf /= sum(pdf)
    return pdf
end

function update(likelihood::Array{Float64}, prior::Array{Float64})
    posterior = prior .* likelihood
    return normalize(posterior)
end

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


# belief = [0.3 0.3 0.1 0.1 0.0 0.1 0.4 0.4 0.1 0.0]
# normalize(belief)
# likelihood = [1. 0. 1. 0.3 0.3 0.1 0.001 0.9 0.1 0.9]
# update(likelihood, belief)
#
# belief = [.05 .05 .05 .05 .55 .05 .05 .05 .05 .05]
# kernel = [.1 .8 .1]
# prior = predict(belief, 1, kernel)
#
# belief = [10.0,100.0,1000.0,5.0,2000.0,200.0,20.0]
# kernel = [1.,2.,3.]
# prior = predict(belief, 0, kernel)
#
# belief = [.05, .05, .05, .05, .55, .05, .05, .05, .05, .05]
# kernel = [.05, .05, .6, .2, .1]
# prior = predict(belief, 3, kernel)

end
