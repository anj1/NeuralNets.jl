function prop(l, x)
	if length(l) == 0 # First layer
		x
	else # Intermediate layers
		l[end].a(l[end] * prop(l[1:end-1], x))
	end
end

# backpropagation;
# with memory for gradients pre-allocated.
# (gradients returned in stor)
function backprop!{T}(net::Vector{T}, stor::Vector{T}, x, t)
	if length(net) == 0 # Final layer
		# Error is simply difference with target
		x .- t
	else # Intermediate layers
		# current layer
		l = net[1]

		# forward activation
		h = l * x
		y = l.a(h)

		# compute error recursively
		δ = l.ad(h) .* backprop!(net[2:end], stor[2:end], y, t)

		# calculate weight and bias gradients
		stor[1].w[:] = δ*x'
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
        h = l * x
        y = l.a(h)
        grad,δ = backprop(net[2:end], y, t)
        δ = l.ad(h) .* δ
        unshift!(grad,NNLayer(δ*x',sum(δ,2),exp,exp))  # Weight gradient
        δ = l.w' * δ
    end
    return grad,δ
end