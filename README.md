# NeuralNets.jl
===============
An open-ended implentation of artificial neural networks in Julia.

Some neat features include:
* Poised to deliver cutting-edge synergy for your business or housecat in real-time!
* Twitter-ready out of the box!
* Both HAL9000 and Skynet proof!
* Low calorie, 100% vegan, and homeopathic friendly!
* Excellent source of vitamin Q!


## Usage
Currently we only have support for multi-layer perceptrons, these are instantiated by using the `MLP(genf,layer_sizes,act)` constructor  to describe the network topology and initialisation procedure as follows:
* `genf::Function` is the function we use to initialise the weights (commonly `rand` or `randn`), 
* `layer_sizes::Vector{Int}` is a vector whose first element is the number of input nodes, and the last element is the number of output nodes, intermediary elements are the numbers of hidden nodes per layer; and 
* `act::Vector{Function}` is the vector of activation functions corresponding to each layer.

For example, `MLP(randn, [4,8,8,2], [relu,logis,ident])` returns a 3-layer network with 4 input nodes, 2 output nodes and two hidden layers with 8 nodes each. The first hidden layer uses a `relu` activation function, the second uses the `logis`, and the output nodes lack any activation function and so we specify them with the `ident` 'function'.

Once the MLP type is constructed we train it using `train()`.

### Activation Functions
There is 'native' support for the following activation functions, if you define an arbitrary activation function, its derivative is calculated automatically using the `ForwardDiff.jl` package. The natively supported activation derivatives, are a bit over twice as fast as derivatives calculated with `ForwardDiff.jl`.
* `ident` the identify function, f(x) = x
* `logis` the logistic sigmoid, f(x) = 1 ./(1 .+ exp(-x)) 
* `logissafe` the logistic sigmoid with a 'safe' derivative which doesn't collapse when evaluating large values of x
* `relu` rectified linear units , f(x) = log(1 .+ exp(x))
* `tanh` hyperbolic tangent as it is defined in regular Julia usage
