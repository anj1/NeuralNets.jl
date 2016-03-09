# NeuralNets.jl
An open-ended implentation of artificial neural networks in Julia.

Some neat features include:
* Poised to deliver cutting-edge synergy for your business or housecat in real-time!
* Twitter-ready out of the box!
* Both HAL9000 and Skynet proof!
* Low calorie, 100% vegan, and homeopathic friendly!
* Excellent source of vitamin Q!

Some less exciting features:
* Flexible network topology with any combination of activation function/layer number.
* Support for a number of common node activation functions in addition to support for arbitrary activation functions with the use of automatic differentiation.
* A broad range of training algorithms to chose from.

Over time we hope to develop this library to encompass more modern types of neural networks, namely deep belief networks.

## Usage
Currently we only have support for multi-layer perceptrons, these are instantiated by using the `MLP(genf,layer_sizes,act)` constructor  to describe the network topology and initialisation procedure as follows:
* `genf::Function` is the function we use to initialise the weights (commonly `rand` or `randn`);
* `layer_sizes::Vector{Int}` is a vector whose first element is the number of input nodes, and the last element is the number of output nodes, intermediary elements are the numbers of hidden nodes per layer;
* `act::Vector{Function}` is the vector of activation functions corresponding to each layer.
* `actd::Vector{Function}` is the vector corresponding to the derivatives of the respective functions in the `act` vector. All of the activation functions provided by NeuralNets have derivatives (which can be seen in the dictionary `NeuralNets.derivs`)

For example, `MLP(randn, [4,8,8,2], [relu,logis,ident], [relud,logisd,identd])` returns a 3-layer network with 4 input nodes, 2 output nodes, and two hidden layers comprised of 8 nodes each. The first hidden layer uses a `relu` activation function, the second uses `logis`. The output nodes lack any activation function and so we specify them with the `ident` 'function'—but this could just as easily be another `logis` to ensure good convergence behaviour on a 1-of-k target vector like you might use with a classification problem.

Once your neural network is initialised (and trained), predictions are made with the `prop(mlp::MLP,x)` command, where `x` is a column vector of the node inputs. Of course `prop()` is also defined on arrays, so inputting a k by n array of data points returns a j by n array of predictions, where k is the number of input nodes, and j is the number of output nodes.

### Activation Functions
There is 'native' support for the following activation functions. If you define an arbitrary activation function its derivative is calculated automatically using the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) package. The natively supported activation derivatives are a bit over twice as fast to evaluate compared with derivatives calculated using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).
* `ident` the identify function, f(x) = x.
* `logis` the logistic sigmoid, f(x) = 1 ./(1 .+ exp(-x)).
* `logissafe` the logistic sigmoid with a 'safe' derivative which doesn't collapse when evaluating large values of x.
* `relu` rectified linear units , f(x) = log(1 .+ exp(x)).
* `tanh` hyperbolic tangent as it is already defined in Julia.

### Training Methods
Once the MLP type is constructed we train it using one of several provided training functions.

* `gdmtrain(nn, x, t)`: This is a natively-implemented gradient descent training algorithm with momentum. Returns (N, L), where N is the trained network and L is the (optional) list of training losses over time. Optional parameters include:
    * `batch_size` (default: n): Randomly selected subset of `x` to use when training extremely large data sets. Use this feature for 'stochastic' gradient descent.
    * `maxiter` (default: 1000): Number of iterations before giving up.
    * `tol` (default: 1e-5): Convergence threshold.
    * `learning_rate` (default: .3): Learning rate of gradient descent. While larger values may converge faster, using values that are too large may result in lack of convergence (you can typically see this happening with weights going to infinity and getting lots of NaNs). It's suggested to start from a small value and increase if it improves learning.
    * `momentum_rate` (default: .6): Amount of momentum to apply. Try 0 for no momentum.
    * `eval` (default: 10): The network is evaluated for convergence every `eval` iterations. A smaller number gives slightly better convergence but each iteration takes a slightly longer time.
    * `store_trace` (default: false): Whether or not to store information on the training state of the network. This information is returned as a list of calculated losses on the entire data set.
    * `show_trace` (default: false): Whether or not to print out information on the training state of the network.
* `adatrain`
* `lmtrain`


## Working Example
