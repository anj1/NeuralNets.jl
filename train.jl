using Optim
import Optim.levenberg_marquardt

loss(y, t) = 0.5 * norm(y .- t).^2

function train{T}(::Type{T}, genf, x, t, hidden_nodes, act, actd, train_method=:momentum_gradient_descent, η=.3, m=.6, c=.0000001)
	# some initializations
	nlayers = length(hidden_nodes) + 1
	n = size(x,2)
	in_dim = size(x,1)
	out_dim = size(t,1)
	nodes = vcat([in_dim],hidden_nodes,[out_dim]) # Buld array of weight matrices' dimensions
	dims = [(nodes[i+1],nodes[i]) for i in 1:nlayers]

	# offsets into the parameter vector
	offs = calc_offsets(T, dims)

	# our single data vector
	buf = genf(offs[end])


	# todo: separate into training and test data
	# todo: make unflatten_net a macro
	# todo: use specified parameters
	# todo: don't discard r; use the parameters as diagnostics

	# train neural net using specified training algorithm.
	# Levenberg-marquardt must be treated as a special case
	# due to the fact that it needs the jacobian.
	if train_method == :levenberg_marquardt
		out_dim==1 || throw("Error: LM only supported with one output neuron.")

		function f(nd)
			vec(prop(unflatten_net(T, vec(nd),  offs, dims, act, actd), x) .- t)
		end

		ln = offs[end]
		buf2 = Array(Float64, ln, n)
		function g(nd)
			for i = 1 : n
				# todo: compute unflattenings just once
				curbuf = pointer_to_array(pointer(buf2)+(i-1)*sizeof(Float64)*ln,(ln,))
				backprop!(unflatten_net(T, vec(nd),   offs, dims, act, actd),
				          unflatten_net(T, curbuf, offs, dims, act, actd), x[:,i], zeros(out_dim))
			end
			buf2'
		end

		r = levenberg_marquardt(f, g, buf)
	else
		function f(nd)
			loss(prop( unflatten_net(T, nd,  offs, dims, act, actd), x), t)
		end

		function g!(nd, ndg)
			backprop!(unflatten_net(T, nd,  offs, dims, act, actd),
			          unflatten_net(T, ndg, offs, dims, act, actd), x,  t)
		end

		r = optimize(f, g!, buf, method=train_method)
	end

	unflatten_net(T, r.minimum, offs, dims, act, actd)
end

# Train a MLP using stochastic gradient decent with momentum.
# l:        array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# eval:     how often we evaluate the loss function (20)
# verbose:  train with printed feedback about the error function (true)
function gdmtrain{T}(net::Vector{T},x,t,η::Real,c::Real,m::Real=0; eval::Int=20, verbose::Bool=true)
    i = e_old = Δ_old = 0
    e_new = loss(prop(net,x),t)
    in_dim,n = size(x)
    converged::Bool = false
    while !converged
        i += 1
        ∇,δ = backprop(net,x,t)
        Δ_new = η*∇ + m*Δ_old  # calculatew Δ weights
        net = net - Δ_new      # update weights                       
        Δ_old = Δ_new           
        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(net,x),t)
            if verbose == true
                println("i: $i\t Loss=$(round(e_new,6))\t ΔLoss=$(round((e_new - e_old),6))\t Avg. Loss=$(round((e_new/n),6))")
            end
        end
        # check for convergence            
        abs(e_new - e_old) < c ? converged = true : nothing
    end
    println("Training converged in less than $i iterations with average error: $(round((e_new/n),4)).")
    println("* learning rate η = $η")
    println("* momentum coefficient m = $m")
    println("* convergence criterion c = $c")
    return net
end