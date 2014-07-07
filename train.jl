using Optim

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
	r=optimize( nd      -> loss(prop( unflatten_net(T, nd,  offs, dims, act, actd), x), t),
	           (nd,ndg) ->  backprop!(unflatten_net(T, nd,  offs, dims, act, actd),
	                                  unflatten_net(T, ndg, offs, dims, act, actd), x,  t),
			   buf,
			   method=train_method)

	unflatten_net(T, r.minimum, offs, dims, act, actd)
end
