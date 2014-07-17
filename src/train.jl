using Optim
import Optim.levenberg_marquardt
using ArrayViews
# these are left in while we fine-tune, resumably we should remove before release


include("lmtrain.jl")
include("gdmtrain.jl")

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

function train{T}(nn_in::T, p::TrainingParams, trainx, valx, traint, valt; verbose::Bool=true, optim_interv=5)
	# todo: separate into training and test data
	# todo: make unflatten_net a macro
	# todo: use specified parameters
	# todo: dont discard r; use the parameters as diagnostics

	# train neural net using specified training algorithm.
	# Levenberg-marquardt must be treated as a special case
	# due to the fact that it needs the jacobian.

	# hooks to call native functions
	if p.train_method == :gdmtrain
		return gdmtrain(nn_in, p, trainx, traint, 10, verbose)
	end

    if p.train_method == :lmtrain
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
	
	if p.train_method == :levenberg_marquardt
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

		optimfunc = :(levenberg_marquardt(nd -> vec(f(nd)), g, nn.buf, tolX=p.c, maxIter=ep_iters))
	else
		function g!(nd, ndg)
			unflatten_net!(nn, nd)
			unflatten_net!(nng, ndg)
			backprop!(nn.net, nng.net, trainx,  traint)
		end

		optimfunc = :(optimize(nd -> 0.5*norm(f(nd)).^2, g!, nn.buf, method=p.train_method, grtol=p.c, iterations=ep_iters, show_trace=verbose))
	end

	converged=false
	numiter=0
	gradnorm=Float64[]
	lastval=Inf
	while !(converged || numiter > p.i)
		# evaluate for a few iterations and look at the results
		r = eval(optimfunc)

		if verbose; println(r.trace); end
		gradnorm = cat(1, gradnorm, [r.trace[i].gradnorm for i = 1 : 10])
		converged = r.x_converged
		numiter += ep_iters

		# now check for validation set convergence
		if length(valx) > 0
			thisval = loss(prop(nn, valx), valt)
			converged = thisval > lastval ? true : converged
			lastval = thisval
		end
	end

	unflatten_net!(nn, r.minimum)
	nn
end

