# Train a MultiLayerPerceptron using gradient decent with momentum.
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function gdmtrain(mlp::MultiLayerPerceptron,
                  x,
                  t;
                  batch_size=size(x,2),
                  maxiter::Int=1000,
                  tol::Real=1e-5,
                  learning_rate=.3,
                  momentum_rate=.6,
                  eval::Int=10,
                  show_trace::Bool=true,
                  store_trace::Bool=false,
                  in_place::Bool=true)
    valid::Bool = false

    η, c, m, b = learning_rate, tol, momentum_rate, batch_size
    i = e_old = e_valid = Δw_old = 0
    e_train = loss(prop(mlp,x),t)
    converged::Bool = false
    params = Dict(["Batch size"=>batch_size,
                   "Max iterations"=>maxiter,
                   "Learning rate"=>learning_rate])
    report = TrainReport("Stochastic Gradient Descent",params)
    display_training_header!(mlp, report)
    diagnostic_trace!(
        report, i, e_train, e_valid, valid, show_trace, in_place, converged)

    n = size(x,2)
    while (!converged && i < maxiter)
        i += 1
        x_batch,t_batch = batch(b,x,t)
        ∇ = backprop(mlp.net,x_batch,t_batch)
        Δw_new = η*∇ .+ m*Δw_old         # calculate Δ weights
        mlp.net = mlp.net .- Δw_new      # update weights
        Δw_old = Δw_new

        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_train
            e_train = loss(prop(mlp,x),t)
            valid && (e_valid = loss(prop(mlp,x),t))
            converged = abs(e_train - e_old) < c # check if converged
            diagnostic_trace!(
                report, i, e_train, e_valid, valid, show_trace, in_place, converged)
        end
    end

    display_training_footer!(report,in_place)
    store_trace ? (return mlp,report) : (return mlp)
end

# Train a MultiLayerPerceptron using Adagrad
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# c:        convergence criterion
# ε:        small constant for numerical stability
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function adatrain(mlp::MultiLayerPerceptron,
                  x,
                  t;
                  batch_size=size(x,2),
                  maxiter::Int=1000,
                  tol::Real=1e-5,
                  learning_rate=.3,
                  lambda=1e-6,
                  eval::Int=10,
                  store_trace::Bool=false,
                  show_trace::Bool=false)

    η, c, λ, b = learning_rate, tol, lambda, batch_size
    i = e_old = Δnet = sumgrad = 0
    e_new = loss(prop(mlp.net,x),t)
    n = size(x,2)
    converged::Bool = false
    e_list = []
    while (!converged && i < maxiter)
        i += 1
        x_batch,t_batch = batch(b,x,t)
        ∇ = backprop(mlp.net,x_batch,t_batch)
        sumgrad += ∇ .^ 2       # store sum of squared past gradients
        Δw = η * ∇ ./ (λ .+ (sumgrad .^ 0.5))   # calculate Δ weights
        mlp.net = mlp.net .- Δw                 # update weights

        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
            converged = abs(e_new - e_old) < c # check if converged
            diagnostic!(
                e_list, e_old, e_new, n, i, store_trace, show_trace,converged)
        end
    end
    convgstr = converged ? "converged" : "didn't converge"
    println("Training $convgstr in less than $i iterations; average error: $(round((e_new/n),4)).")
    println("* learning rate η = $η")
    println("* convergence criterion c = $c")
    return mlp, e_list
end
