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
hidden_nodes = [3, 3]
act   = [relu,  relu,  logis]
actd  = [relud, relud, logisd]

L = train(NNLayer,
		  randn,
		  x,
		  t,
		  hidden_nodes,
		  act,
		  actd,
		  :momentum_gradient_descent)

@show prop(L, x)