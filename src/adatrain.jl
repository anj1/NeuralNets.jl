# Train a MLP using Adagrad
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# c:        convergence criterion
# ε:        small constant for numerical stability
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function adatrain(mlp::MLP, p::TrainingParams, x, t, eval::Int=10, verbose::Bool=true)
    η, c, λ = p.η, p.c, 1e-6
    i = e_old = Δ = ∑ = 0
    e_new = loss(prop(mlp.net,x),t)
    n = size(x,2)
    converged::Bool = false
    while (!converged && i < p.i) # while not converged and i less than maxiter
        i += 1
        # Start of the update step
        # eventually the update step here should be the sole definition of gdmtrain()
        # and we can roll all of the ancillary things in to train()
        # to economise on code to handle convergence notifications
        # flexibility with verification sets, etc...
        ∇,δ = backprop(mlp.net,x,t)

        ∇ .* ∇
        ∑ += ∇ .* ∇
        ∇_adj = ∇ ./ (λ .+ sqrt(∑))

        Δ = η * ∇_adj # calculate Δ weights

        # End of the update calculation step
        mlp.net = mlp.net .- Δ      # update weights
        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
            converged = abs(e_new - e_old) < c # check if converged
        end
        if verbose && i % 100 == 0
            println("i: $i\tLoss=$(round(e_new,6))\tΔLoss=$(round((e_new - e_old),6))\tAvg. Loss=$(round((e_new/n),6))")
        end
    end
    convgstr = converged ? "converged" : "didn't converge"
    println("Training $convgstr in less than $i iterations; average error: $(round((e_new/n),4)).")
    println("* learning rate η = $η")
    println("* convergence criterion c = $c")
    return mlp
end
