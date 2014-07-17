# Train a MLP using Levenberg-Marquardt optimisation.
function lmtrain(mlp::MLP,
                 x,
                 t;
                 iterations::Int=1000,
                 tol::Real=1e-5,
                 learning_rate=.3,
                 momentum_rate=.6,
                 eval::Int=10,
                 verbose::Bool=true)

    minλ = 1e16     # default values curtesy of Optim.jl
    maxλ = 1e-16
    λ    = 100

    # todo: make this thread-safe
    nn  = deepcopy(nn_in)
    nng = deepcopy(nn)

    ln = nn.offs[end]
    n = size(x,2)
    jacobian = Array(Float64, ln, n)

    function g(nd)                      # generate the jacobian
        unflatten_net!(nn, vec(nd))
        for i = 1 : n
            jacobcol = view(jacobian, :, i)
            unflatten_net!(nng, jacobcol)
            backprop!(nn.net, nng.net, x[:,i], zeros(out_dim))
        end
        return jacobian'
    end

    i = 0
    e_old = loss(prop(w_temp,x),t)

    # r = levenberg_marquardt(nd -> vec(f(nd)), g, nn.buf, tolX=p.c, maxIter=p.i)
    while  (!converged && i < iterations) # while not converged and i less than maxiter
        i += 1

        J = g(x) # calculate hessian
        H = J'*J # linear hessian approximation
        
        Δw = sum(inv(H + λ*diagm(diag(H)))*J,3) # axis 3 will be for each data point

        # need some code in here to make Δw of type Array{NNLayer} 

        w_temp = mlp.net - Δw       # tentatively update weights
        e_new=loss(prop(w_temp,x),t)

        if e_new - e_old < 0        # if new step decreases loss
            λ = max(0.1*λ, minλ)    # decrease λ
            mlp.net = w_temp        # update weights
            e_old = e_new           # prepare for another iteration
        else
            λ = min(10*λ, maxλ)     # increase λ
        end

        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
            converged = abs(e_new - e_old) < c # check if converged
        end
    end
    convgstr = converged ? "converged" : "didn't converge"
    println("Training $convgstr in less than $i iterations; average error: $(round((e_new/n),4)).")    
    return mlp,converged
end