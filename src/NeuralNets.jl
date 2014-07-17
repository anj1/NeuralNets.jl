module NeuralNets

using Optim
using ArrayViews

import Optim:levenberg_marquardt

# functions
export train, gdmtrain, prop
export logis, logisd, logissafe, logissafed, relu, relud, ident, identd, tanhd

# types
export MLP

# types
#include("types.jl")

# multi-layer perceptrons
include("activations.jl")
include("mlp.jl")

# training
include("backprop.jl")
include("lmtrain.jl")
include("gdmtrain.jl")
include("train.jl")

end
