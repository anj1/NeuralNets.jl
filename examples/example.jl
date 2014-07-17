# bring in definitions
require("types.jl")
require("activations.jl")
require("mlp.jl")
require("backprop.jl")
require("train.jl")

# xor training data
x = [
    0.0 1.0 0.0 1.0
    0.0 0.0 1.0 1.0
    ]

t = [
    0.0 1.0 1.0 0.0
    ]

# network topology
ind = size(x,1)
outd = size(t,1)

layer_sizes = [ind, 3, 3, outd]
act   = [relu,  relu,  logis]

# initialize net
mlp = MLP(randn, layer_sizes, act)

# training parameters
p = TrainingParams(1000, 1e-5, .3, .6, :levenberg_marquardt)

# self-explanatory
mlp1 = train(mlp, p, x, t)
@show prop(mlp1, x)

mlp2 = gdmtrain(mlp, p, x, t)
@show prop(mlp2, x)

