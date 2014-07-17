using Optim
import Optim.levenberg_marquardt
using ArrayViews
# these are left in while we fine-tune, resumably we should remove before release


include("lmtrain.jl")
include("gdmtrain.jl")

loss(y, t) = 0.5 * norm(y .- t).^2


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

		r = levenberg_marquardt(nd -> vec(f(nd)), g, nn.buf, tolX=p.c, maxIter=p.i)
	else
		function g!(nd, ndg)
			unflatten_net!(nn, nd)
			unflatten_net!(nng, ndg)
			backprop!(nn.net, nng.net, x,  t)
		end

		r = optimize(nd -> 0.5*norm(f(nd)).^2, g!, nn.buf, method=p.train_method, grtol=p.c, iterations=p.i, show_trace=verbose)
	end

	unflatten_net!(nn, r.minimum)
	nn
end

