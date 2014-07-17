using NeuralNets

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

# initialize net
mlp = MLP(randn, layer_sizes, act)

# train without a validation set
mlp1 = train(mlp, x, [], t, [], train_method=:levenberg_marquardt)
@show prop(mlp1, x)

mlp2 = gdmtrain(mlp, x, t)
@show prop(mlp2, x)

