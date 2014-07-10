using Optim
import Optim.levenberg_marquardt
using ArrayViews

loss(y, t) = 0.5 * norm(y .- t).^2

type TrainingParams
    i::Int              # iterations for convergence
    c::Real             # convergence criterion
    η::Real             # learning rate
    m::Real             # momentum amount
    train_method        # training method
end

function Base.show(io::IO, p::TrainingParams)
    print(io, summary(p),"\n")
    print(io, "Parameters for training a neural network:","\n")
    print(io, "* maximum iterations: ", p.i,"\n")
    print(io, "* convergence criterion: ", p.c,"\n")
    print(io, "* learning rate: ", p.η,"\n")
    print(io, "* momentum coefficient: ", p.m,"\n")
    print(io, "* train method: ", p.train_method,"\n")
end

function train{T}(nn_in::T, p::TrainingParams, x, t; verbose::Bool=true)
	# todo: separate into training and test data
	# todo: make unflatten_net a macro
	# todo: use specified parameters
	# todo: don't discard r; use the parameters as diagnostics

	# train neural net using specified training algorithm.
	# Levenberg-marquardt must be treated as a special case
	# due to the fact that it needs the jacobian.

	# hooks to call native functions
	if p.train_method == :gdmtrain
		return gdmtrain(nn_in, p, x, t, 10, verbose)
	end

	# todo: make this thread-safe
	nn  = deepcopy(nn_in)
	nng = deepcopy(nn)

	function f(nd)
		unflatten_net!(nn, vec(nd))
		prop(nn.net, x).-t
	end
	
	if p.train_method == :levenberg_marquardt
		out_dim = size(t,1)
		out_dim==1 || throw("Error: LM only supported with one output neuron.")

		ln = nn.offs[end]
		n = size(x,2)
		jacobian = Array(Float64, ln, n)
		function g(nd)
			unflatten_net!(nn, vec(nd))
			for i = 1 : n
				jacobcol = view(jacobian, :, i)
				unflatten_net!(nng, jacobcol)
				backprop!(nn.net, nng.net, x[:,i], zeros(out_dim))
			end
			jacobian'
		end

		r = levenberg_marquardt(nd -> vec(f(nd)), g, nn.buf)
	else
		function g!(nd, ndg)
			unflatten_net!(nn, nd)
			unflatten_net!(nng, ndg)
			backprop!(nn.net, nng.net, x,  t)
		end

		r = optimize(nd -> 0.5*norm(f(nd).^2), g!, nn.buf, method=p.train_method, grtol=p.c, iterations=p.i, show_trace=verbose)
	end

	unflatten_net!(nn, r.minimum)
	nn
end

# Train a MLP using stochastic gradient decent with momentum.
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
function gdmtrain(mlp::MLP, p::TrainingParams, x, t; eval::Int=10, verbose::Bool=true)
    η, c, m = p.η, p.c, p.m
    i = e_old = Δw_old = 0
    e_new = loss(prop(mlp.net,x),t)
    n = size(x,2)
    converged::Bool = false
    while !converged
        i += 1
        # Start of the update step
        # eventually the update step here should be the sole definition of gdmtrain()
        # and we can roll all of the ancillary things in to train()
        # to economise on code to handle convergence notifications
        # flexibility with verification sets, etc...
        ∇,δ = backprop(mlp.net,x,t)
        Δw_new = η*∇ .+ m*Δw_old  # calculate Δ weights
        # End of the update calculation step            
        mlp.net = mlp.net .- Δw_new      # update weights                       
        Δw_old = Δw_new 
        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
            converged = abs(e_new - e_old) < c # check if converged
        end
        if verbose && i % 100 == 0
                println("i: $i\t Loss=$(round(e_new,6))\t ΔLoss=$(round((e_new - e_old),6))\t Avg. Loss=$(round((e_new/n),6))")
        end        
        i >= p.i && break # check if hit the max iterations limit 
    end
    convgstr = converged ? "converged" : "didn't converge"
    println("Training $convgstr in less than $i iterations; average error: $(round((e_new/n),4)).")
    println("* learning rate η = $η")
    println("* momentum coefficient m = $m")
    println("* convergence criterion c = $c")
    return mlp
end

# Train a MLP using Levenberg-Marquardt optimisation.
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# λ:        mixing coefficient, large λ -> gradient descent, small λ -> quadratic approximated hessian
# i:        number of accepted steps
# j:        number of attempts at finding steps
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
# TODO need to calculate hessian approximation H
function lmtrain(mlp::MLP, p::TrainingParams, x, t; eval::Int=10, verbose::Bool=true)
    η, c, m = p.η, p.c, p.m
    i = e_old = Δ_old = 0
    λ = 1e-2
    converged::Bool = false    
    while !converged
        i += 1          
        # Start of the update step
        H = [kron(d[i],d[i]) for i in length(d)] 
        while true 
            j += 1 
            ∇,δ = backprop(mlp.net,x,t)
            Δw = inv(H .+ λ*diagm(diag(H))) * ∇        # caclulate new setp
            e_new = loss(prop(mlp.net - Δw_new,x),t)    # test new update
            Δe = e_new - e_old
            if Δe > 0           
                λ *= 10   # if new step -> error goes up, increase λ and try again
            else                                         
                λ /= 10   # if new step -> error goes down, decrease λ
                break
            end
            if j - i > 1e4
                println("Oh god something's fucked up here")
                println("j= $j, i= $i")
                break
            end
        end
        # End of the update calculation step            
        mlp.net = mlp.net .- Δw_new     # accept step, update weights               
        Δw_old = Δw_new
        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
            converged = abs(e_new - e_old) < c # check if converged
        end
        if i % 100 == 0 && verbose == true
            println("i: $i\t Loss=$(round(e_new,6))\t ΔLoss=$(round((e_new - e_old),6))\t Avg. Loss=$(round((e_new/n),6))")
        end        
        i >= p.i && break # check if hit the max iterations limit 
    end
end