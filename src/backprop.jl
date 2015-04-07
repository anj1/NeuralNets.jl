loss(y, t) = 0.5 * norm(y .- t).^2 

function prop(net, x)
	if length(net) == 0 # First layer
		x
	else # Intermediate layers
		net[end].a(applylayer(net[end], prop(net[1:end-1], x)))
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
function backprop!{T}(net::Vector{T}, stor::Vector{T}, x, t, inplace)
	if length(net) == 0 # Final layer
		# Error is simply difference with target
		r = x .- t
		r[isnan(r)]=0
		r
	else # Intermediate layers
		# current layer
		l = net[1]

		# forward activation
		h = applylayer(l, x)
		y = l.a(h)

		# compute error recursively
		δ = l.ad(h) .* backprop!(net[2:end], stor[2:end], y, t, inplace)

		# KL divergence penalty for sparsity constraint
		if l.sparse
			β  = l.sparsecoef
			ρ  = l.sparsity
			pm = vec(mean(y,2))
			δ += β*((1-ρ)./(1.-pm) - ρ./pm)
		end

		# calculate weight and bias gradients
		if inplace
			stor[1].w[:] = vec(δ*x')
			stor[1].b[:] = sum(δ,2)
		else
			stor[1].w    =     δ*x'
			stor[1].b    = vec(sum(δ,2))
		end

		# propagate error
		l.w' * δ
	end
end

backprop!{T}(net::Vector{T}, stor::Vector{T}, x, t) = backprop!(net, stor, x, t, true)

function backprop{T}(net::Vector{T}, x, t)
	stor = [copy(l) for l in net]
	backprop!(net, stor, x, t, false)
	stor
end