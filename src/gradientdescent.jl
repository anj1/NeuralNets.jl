# batch
# function to retrieve a random subset of data
# currently quite ugly, if anyone knows how to do this better go ahead
function batch(b::Int,x::Array,t::Array)
    n = size(x,2)
    b == n && return x,t
    b > n && throw("Error: Batch size larger than the number of data points supplied.")
    index = rand(1:n, b)
    return x[:,index],t[:,index]
end

# store/show trace of loss for diagnostic purposes
function diagnostic!(e_list, e_old, e_new, n, i, store_trace, show_trace)
    if store_trace
        push!(e_list, e_new)
    end
    if show_trace
        println("i: $i\tLoss=$(round(e_new,6))\tΔLoss=$(round((e_new - e_old),6))\tAvg. Loss=$(round((e_new/n),6))")
    end
end

# Train a MLP using gradient decent with momentum.
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function gdmtrain(mlp::MLP,
                  x,
                  t;
                  batch_size=size(x,2),
                  maxiter::Int=1000,
                  tol::Real=1e-5,
                  learning_rate=.3,
                  momentum_rate=.6,             
                  eval::Int=10,
                  store_trace::Bool=false,
                  show_trace::Bool=false)
    n = size(x,2)
    η, c, m, b = learning_rate, tol, momentum_rate, batch_size
    i = e_old = Δw_old = 0
    e_new = loss(prop(mlp.net,x),t)
    converged::Bool = false

    e_list = []
    while (!converged && i < maxiter)
        i += 1
        x_batch,t_batch = batch(b,x,t)
        ∇ = backprop(mlp.net,x_batch,t_batch)
        Δw_new = η*∇ .+ m*Δw_old         # calculate Δ weights   
        mlp.net = mlp.net .- Δw_new      # update weights                       
        Δw_old = Δw_new 

        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
            converged = abs(e_new - e_old) < c # check if converged
            diagnostic!(e_list, e_old, e_new, n, i, store_trace, show_trace)
        end
    end
    convgstr = converged ? "converged" : "didn't converge"
    println("Training $convgstr in less than $i iterations; average error: $(round((e_new/n),4)).")
    println("* learning rate η = $η")
    println("* momentum coefficient m = $m")
    println("* convergence criterion c = $c")
    return mlp, e_list
end

# Train a MLP using Adagrad
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# c:        convergence criterion
# ε:        small constant for numerical stability
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function adatrain(mlp::MLP,
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
            diagnostic!(e_list, e_old, e_new, n, i, store_trace, show_trace)
        end
    end
    convgstr = converged ? "converged" : "didn't converge"
    println("Training $convgstr in less than $i iterations; average error: $(round((e_new/n),4)).")
    println("* learning rate η = $η")
    println("* convergence criterion c = $c")
    return mlp, e_list
end
