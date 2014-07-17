# Train a MLP using Levenberg-Marquardt optimisation.


function lmtrain(mlp::MLP, p::TrainingParams, x, t, eval::Int=10, verbose::Bool=true)
    # todo: make this thread-safe
    nn  = deepcopy(nn_in)
    nng = deepcopy(nn)

    function f(nd)                      # objective function to minimise
        unflatten_net!(nn, vec(nd))
        prop(nn.net, x).-t
    end
    
    out_dim = size(t,1)
    out_dim==1 || throw("Error: LM only supported with one output neuron.")

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

    # r = levenberg_marquardt(nd -> vec(f(nd)), g, nn.buf, tolX=p.c, maxIter=p.i)
    while !converged
        i += 1

        # Levenberg-Marquardt update
        J = g(x) # calculate hessian
        H = J'*J # linear hessian approximation
        
        Δw = -inv(H + sqrt(λ)*diagm(diag(H)))*J
        # need some code in here to turn Δw into an Array{NNLayer} 


        if loss(prop(mlp.net-Δw,t) < 0
            λ = max(0.1*λ, minλ)
        else
            λ = min(10*λ, maxλ)
        end



        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
            converged = abs(e_new - e_old) < c # check if converged
    end

    J = g(x)


    unflatten_net!(nn, r.minimum)
    nn
end