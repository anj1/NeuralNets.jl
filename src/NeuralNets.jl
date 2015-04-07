module NeuralNets

using Optim
using ArrayViews

import Optim:levenberg_marquardt
import Base: show


# functions
export train, gdmtrain, adatrain, prop
export logis, logisd, logissafe, logissafed, relu, relud, ident, identd, tanhd

# types
export MLP, NNLayer

# multi-layer perceptrons
include("train.jl")
include("activations.jl")
#include("losses.jl")
include("mlp.jl")

# training
include("backprop.jl")
include("gradientdescent.jl")
#include("lmtrain.jl")
end
