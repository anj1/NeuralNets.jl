using Optim
import Optim.levenberg_marquardt
using ArrayViews

loss(y, t) = 0.5 * norm(y .- t).^2

# possibly useful functions to diagnose convergence problems 
# possibly to suggest learning rates with a while loop checking 
# if the first step produces a NaN in any of the weights
function Base.isnan(net::Array{NNLayer})
    nans = 0 
    for l in net
        nans += sum(isnan(l.w)) + sum(isnan(l.b))
    end
    return nans > 0
end
Base.isnan(mlp::MLP) = isnan(mlp.net)

function train{T}(nn_in::T,
	              trainx,
	              valx,
	              traint,
	              valt;
	              maxiter::Int=1000,
	              tol::Real=1e-5,
	              learning_rate=.3,
	              momentum_rate=.6,
	              train_method=:gradient_descent,
	              verbose::Bool=true,
	              ep_iter::Int=5)

	# todo: separate into training and test data
	# todo: make unflatten_net a macro
	# todo: use specified parameters
	# todo: dont discard r; use the parameters as diagnostics

	# train neural net using specified training algorithm.
	# Levenberg-marquardt must be treated as a special case
	# due to the fact that it needs the jacobian.

	# hooks to call native functions
	if train_method == :gdmtrain
		return gdmtrain(nn_in, trainx, traint, maxiter, tol, learning_rate, momentum_rate, eval, verbose)
	end

    if train_method == :lmtrain
        return nothing
        # return lmtrain(nn_in, stuff)
    end

	# todo: make this thread-safe
	nn  = deepcopy(nn_in)
	nng = deepcopy(nn)

	function f(nd)
		unflatten_net!(nn, vec(nd))
		prop(nn.net, trainx).-traint
	end
	
	converged=false
	numiter=0
	gradnorm=Float64[]
	lastval=Inf
	r = []
	if train_method == :levenberg_marquardt
		out_dim = size(traint,1)
		out_dim==1 || throw("Error: LM only supported with one output neuron.")

		ln = nn.offs[end]
		n = size(trainx,2)
		jacobian = Array(Float64, ln, n)
		function g(nd)
			unflatten_net!(nn, vec(nd))
			for i = 1 : n
				jacobcol = view(jacobian, :, i)
				unflatten_net!(nng, jacobcol)
				backprop!(nn.net, nng.net, trainx[:,i], zeros(out_dim))
			end
			jacobian'
		end

		while numiter <= maxiter
			r = levenberg_marquardt(nd -> vec(f(nd)), g, nn.buf, tolX=tol, maxIter=ep_iter)
			
			numiter += ep_iter

			lastval, vc = convg_check(r, nn, valx, valt, lastval)
			if vc || r.x_converged; break; end
		end
	else
		function g!(nd, ndg)
			unflatten_net!(nn, nd)
			unflatten_net!(nng, ndg)
			backprop!(nn.net, nng.net, trainx,  traint)
		end

		while numiter <= maxiter
			r = optimize(nd -> 0.5*norm(f(nd)).^2, g!, nn.buf, method=train_method, grtol=tol, iterations=maxiters)

			gradnorm = proc_results(r, gradnorm, verbose, ep_iter)
			numiter += ep_iter

			lastval, vc = convg_check(r, nn, valx, valt, lastval)
			if vc || r.x_converged; break; end
		end
	end

	unflatten_net!(nn, r.minimum)
	nn
end

function proc_results(r, gradnorm, verbose, ep_iter)
	if verbose; println(r.trace[ep_iter]); end
	gradnorm = cat(1, gradnorm, [r.trace[i].gradnorm for i = 1 : ep_iter])
	gradnorm
end

function convg_check(r, nn, valx, valt, lastval)
	converged = false
	if length(valx) > 0
		thisval = loss(prop(nn, valx), valt)
		converged = thisval > lastval ? true : converged
		lastval = thisval
	end
	lastval, converged
end