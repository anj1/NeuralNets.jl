function prop(net, x)
	if length(net) == 0 # First layer
		x
	else # Intermediate layers
		net[end].a(net[end] * prop(net[1:end-1], x))
	end
end

prop(mlp::MLP,x) = prop(mlp.net,x)

# add some 'missing' functionality to ArrayViews
function setindex!{T}(dst::ContiguousView, src::Array{T}, idx::UnitRange)
	offs = dst.offset
	dst.arr[offs+idx.start:offs+idx.stop] = src
end

# backpropagation;
# with memory for gradients pre-allocated.
# (gradients returned in stor)
function backprop!{T}(net::Vector{T}, stor::Vector{T}, x, t)
	if length(net) == 0 # Final layer
		# Error is simply difference with target
		r = x .- t
		r[find(isnan(r))]=0
		r
	else # Intermediate layers
		# current layer
		l = net[1]

		# forward activation
		h = l * x
		y = l.a(h)

		# compute error recursively
		δ = l.ad(h) .* backprop!(net[2:end], stor[2:end], y, t)

		# calculate weight and bias gradients
		stor[1].w[:] = vec(δ*x')
		stor[1].b[:] = sum(δ,2)

		# propagate error
		l.w' * δ
	end
end

# backprop(net,x,t) returns array of gradients and error for net 
# todo: make gradient unshift! section more generic
function backprop{T}(net::Vector{T}, x, t)
    if length(net) == 0   	# Final layer
        δ  = x .- t     	# Error (δ) is simply difference with target
        grad = T[]        	# Initialize weight gradient array
    else                	# Intermediate layers
        l = net[1]
        h = l * x           # Not a typo!
        y = l.a(h)
        grad,δ = backprop(net[2:end], y, t)
        δ = l.ad(h) .* δ
        unshift!(grad,NNLayer(δ*x',vec(sum(δ,2)),exp,exp))  # Weight gradient
        δ = l.w' * δ
    end
    return grad,δ
end