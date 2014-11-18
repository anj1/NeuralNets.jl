module NeuralNets

using Optim
using ArrayViews

import Optim:levenberg_marquardt
import Base: show, push!


# functions
export train, gdmtrain, adatrain, prop
export logis, logisd, logissafe, logissafed, relu, relud, ident, identd, tanhd

# types
export MultiLayerPerceptron, NNLayer

# multi-layer perceptrons
include("activations.jl")
include("losses.jl")
include("mlp.jl")

# training
include("backprop.jl")
include("train.jl")
include("optim_train.jl")
include("gradientdescent.jl")
include("lmtrain.jl")
end
