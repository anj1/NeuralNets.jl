# bring in definitions
require("activ.jl")
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
layer_sizes = [2, 3, 3, 1]
act   = [relu,  relu,  logis]
actd  = [relud, relud, logisd]

# initialize net
mlp = MLP(randn, layer_sizes, act, actd)

# training parameters
p = TrainingParams(1000, 1e-7, .3, .6, :levenberg_marquardt)

# self-explanatory
# Train with external LM method
mlp1 = train(mlp, p, x, t)
@show prop(mlp1, x)

# Now try training with native GDM method
p.train_method = :gdmtrain

mlp2 = train(mlp, p, x, t)
@show prop(mlp2, x)