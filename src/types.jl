# types for indicating the training state
type TrainState
    iter::Int
    loss::Vector{Real}
end

function TrainState(i::Integer, loss::Real)
    TrainState(int(i), [loss])
end

function TrainState(i::Int, loss::Real, validloss::Real)
    TrainState(i, [loss,validloss])
end

function Base.show(io::IO, t::TrainState)
    if length(t.loss) == 1
        @printf io "i: %6d   %14e\n" t.iter t.loss[1]
    else
        @printf io "i: %6d   %14e  %14e\n" t.iter t.loss[1] t.loss[2]
    end
end

type TrainResult
    converged::Bool    
    method::ASCIIString
    iter::Int
    maxiter::Int
    loss::Vector{Real}
    misc::Dict
end

function Base.show(io::IO, result::TrainResult)
    @printf method
    @printf iter
    @printf maxiter
    @printf converged
    if !isempty(result.misc)
        for (key, value) in t.misc
            @printf io " * %s: %s\n" key value
        end
    end
end