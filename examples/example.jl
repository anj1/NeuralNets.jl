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
act = [relu,  relu,  logis]

# initialize net
mlp = MultiLayerPerceptron(randn, layer_sizes, act)
gdmtrain(mlp, x, t, x, t; store_trace=true, in_place=false)