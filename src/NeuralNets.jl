module NeuralNets

using Optim
using ArrayViews

import Optim:levenberg_marquardt

# functions
export train, prop
export logis, logisd, logissafe, logissafed, relu, relud, ident, identd, tanhd

# types
export MLP

# types
include("types.jl")

# multi-layer perceptrons
include("activations.jl")
include("mlp.jl")

# training
include("lmtrain.jl")
include("gdmtrain.jl")
include("train.jl")

end
