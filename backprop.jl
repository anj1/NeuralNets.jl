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
function backprop!{T}(l::Vector{T}, stor::Vector{T}, x, t)
	if length(l) == 0 # Final layer
		# Error is simply difference with target
		x .- t
	else # Intermediate layers
		# current layer
		head = l[1]

		# forward activation
		h = head * x
		y = head.a(h)

		# compute error recursively
		δ = head.ad(h) .* backprop!(l[2:end], stor[2:end], y, t)

		# calculate weight and bias gradients
		stor[1].w[:] = δ*x'
		stor[1].b[:] = sum(δ,2)

		# propagate error
		head.w' * δ
	end
end

# Given a flattened vector (buf), and a list of
# offsets into that vector (for each layer), unflatten
# the vector into a list of NN layers.
# parms is a list of net-specific parameter tuples.
# act and actd are a list of activation functions
function unflatten_net{NNType}(::Type{NNType}, buf::Vector, offs, parms, act, actd)
	nlayers = length(parms)
	L = Array(NNType,nlayers)
	pbuf = pointer(buf)
	for i = 1 : nlayers
		toff = i > 1 ? offs[i-1] : 0
		L[i] = NNType(pbuf + toff*sizeof(eltype(buf)),
		              parms[i], act[i], actd[i])
	end
	L
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