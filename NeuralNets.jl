module NeuralNets
    using Optim
    using ArrayViews

    import Optim.levenberg_marquardt

    export # put things here

    # types
    include("types.jl")

    # training
    include("lmtrain.jl")
    include("gdmtrain.jl")
    include("train.jl")

    # multi-layer perceptrons
    include("activations.jl")
    include("mlp.jl")
end
