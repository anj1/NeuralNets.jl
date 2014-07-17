
type TrainingParams
    i::Int              # iterations for convergence
    c::Real             # convergence criterion
    η::Real             # learning rate
    m::Real             # momentum amount
    train_method        # training method
end

function Base.show(io::IO, p::TrainingParams)
    print(io, summary(p),"\n")
    print(io, "Parameters for training a neural network:","\n")
    print(io, "* maximum iterations: ", p.i,"\n")
    print(io, "* convergence criterion: ", p.c,"\n")
    print(io, "* learning rate: ", p.η,"\n")
    print(io, "* momentum coefficient: ", p.m,"\n")
    print(io, "* train method: ", p.train_method,"\n")
end
