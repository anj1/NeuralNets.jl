# NeuralNets.jl
An open-ended implentation of artificial neural networks in Julia.

Some neat features include:
* Poised to deliver cutting-edge synergy for your business or housecat in real-time!
* Twitter-ready out of the box!
* Both HAL9000 and Skynet proof!
* Low calorie, 100% vegan, and homeopathic friendly!
* Excellent source of vitamin Q!


## Usage
Currently we only have support for multi-layer perceptrons, these are instantiated by using the `MLP(genf,layer_sizes,act)` constructor  to describe the network topology and initialisation procedure as follows:
* `genf::Function` is the function we use to initialise the weights (commonly `rand` or `randn`); 
* `layer_sizes::Vector{Int}` is a vector whose first element is the number of input nodes, and the last element is the number of output nodes, intermediary elements are the numbers of hidden nodes per layer;
* `act::Vector{Function}` is the vector of activation functions corresponding to each layer.

For example, `MLP(randn, [4,8,8,2], [relu,logis,ident])` returns a 3-layer network with 4 input nodes, 2 output nodes, and two hidden layers comprised of 8 nodes each. The first hidden layer uses a `relu` activation function, the second uses `logis`, and the output nodes lack any activation function and so we specify them with the `ident` 'function'.

### Activation Functions
There is 'native' support for the following activation functions. If you define an arbitrary activation function its derivative is calculated automatically using the `ForwardDiff.jl` package. The natively supported activation derivatives are a bit over twice as fast to evaluate compared with derivatives calculated using `ForwardDiff.jl`.
* `ident` the identify function, f(x) = x.
* `logis` the logistic sigmoid, f(x) = 1 ./(1 .+ exp(-x)).
* `logissafe` the logistic sigmoid with a 'safe' derivative which doesn't collapse when evaluating large values of x.
* `relu` rectified linear units , f(x) = log(1 .+ exp(x)).
* `tanh` hyperbolic tangent as it is already defined in Julia.

### Training Methods
Once the MLP type is constructed we train it using one of several provided training functions.

* `train(nn, trainx, valx, traint, valt)`: This training method relies on calling the external [Optim.jl](https://github.com/JuliaOpt/Optim.jl) package. By default it uses the `gradient_descent` algorithm. However, by setting the `train_method` parameter, the following algorithms can also be selected: `levenberg_marquardt`, `momentum_gradient_descent`, or `nelder-mead`. The function accepts two data sets: the training data set (inputs and outputs given with `trainx` and `traint`) and the validation set (`valx`, `valt`). Input data must be a matrix with each data point occuring as a column of the matrix. Optional parameters include:
    * `maxiter` (default: 100): Number of iterations before giving up.
    * `tol` (default: 1e-5): Convergence threshold. Does not affect `levenberg_marquard`.
    * `ep_iterl` (default: 5): Performance is evaluated on the validation set every `ep_iter` iterations. A smaller number gives slightly better convergence but each iteration takes a slightly longer time.
    * `verbose` (default: true): Whether or not to print out information on the training state of the network.

* `gdmtrain(nn, x, t)`: This is a natively-implemented gradient descent training algorithm with momentum. Optional parameters include:
    * `maxiter` (default: 1000): Number of iterations before giving up.
    * `tol` (default: 1e-5): Convergence threshold.
    * `learning_rate` (default: .3):
    * `momentum_rate` (default: .6): amount of momentum to apply. Try 0 for no momentum.
    * `eval` (default: 10): The network is evaluated for convergence every `eval` iterations. A smaller number gives slightly better convergence but each iteration takes a slightly longer time.
    * `verbose` (default: true): Whether or not to print out information on the training state of the network.
* `adatrain`
* `lmtrain`


## Working Example