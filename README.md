NeuralNets.jl
===============
An open-ended implentation of artificial neural networks in Julia. 

Currently we only have support for multi-layer perceptrons, these are instantiated by using the `MLP()` constructor as follows: `MLP(genf,layer_sizes,act)`, where 
* `genf::Function` is the function we use to initialise the weights. 
* `layer_sizes::Vector{Int}` is a vector whose first element is the number of input nodes, and the last element is the number of output nodes, intermediary elements are the numbers of hidden nodes per layer; and 
* `act::Vector{Function}` is the vector of activation functions corresponding to each layer.


For example, `MLP(randn,[4,8,8,2],[relu,relu,logis])` instantiates a 3-layer network 4 input nodes, and 2 output nodes. 


# network topology
ind = size(x,1)
outd = size(t,1)

layer_sizes = [ind,3,3,outd]
act   = [relu,relu,logis]

# initialize net
mlp = MLP(randn, layer_sizes, act)

# training parameters
p = TrainingParams(1000, 1e-5, .3, .6, :levenberg_marquardt)

# self-explanatory
mlp1 = train(mlp, p, x, t)
@show prop(mlp1, x)

mlp2 = gdmtrain(mlp, p, x, t)
@show prop(mlp2, x)


