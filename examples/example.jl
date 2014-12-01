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
actd = [relud, relud, logisd]

# initialize net
mlp = MLP(randn, layer_sizes, act, actd)
gdmtrain(mlp, x, t, x, t; show_trace=true)