module NeuralNets

using Optim
using ArrayViews

import Optim:levenberg_marquardt

# functions
export train, gdmtrain, adatrain, prop
export logis, logisd, logissafe, logissafed, relu, relud, ident, identd, tanhd

# types
export MLP, NNLayer, Filter1D, ShiftFilterBank

# multi-layer perceptrons
include("activations.jl")
include("losses.jl")
include("mlp.jl")
include("lcnn.jl")

# training
include("backprop.jl")
include("lmtrain.jl")
include("gradientdescent.jl")
include("train.jl")

end
