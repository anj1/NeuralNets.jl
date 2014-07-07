# Types and function definitions for multi-layer perceptrons

type NNLayer{T}
	w::Matrix{T}
	b::Vector{T}
	a::Function
	ad::Function
end

# Construct an NNLayer type from pre-existing data,
# keeping the references.
function NNLayer{T}(buf::Ptr{T}, dims::(Int,Int), a::Function, ad::Function)
	w = pointer_to_array(buf, dims)
	b = pointer_to_array(buf + sizeof(T)*prod(dims), (dims[1],))
	NNLayer(w,b,a,ad)
end

# In all operations between two NNLayers, the activations functions are taken from the first NNLayer
*(l::NNLayer, x::Array) = l.w*x .+ l.b

# in flattened representation, these aren't needed.
# .*(c::Number, l::NNLayer) = NNLayer(c*l.w, c*l.b, l.a, l.ad)            
# -(l::NNLayer, m::NNLayer) = NNLayer(l.w - m.w, l.b - m.b, l.a, l.ad)    
# -(l::NNLayer, c::Number) = NNLayer(l.w .- c, l.b .- c, l.a, l.ad)       
# +(l::NNLayer, m::NNLayer) = NNLayer(l.w + m.w, l.b + m.b, l.a, l.ad)  
# +(l::NNLayer, c::Number) = NNLayer(l.w .+ c, l.b .+ c, l.a, l.ad)   

# For the NNLayer type, given a set of layer dimensions,
# compute offsets into the flattened vector.
# We only have to do this once, when a net is initialized.
function calc_offsets(::Type{NNLayer}, dims)
	nlayers = length(dims)
	offs = Array(Int,nlayers)
	sumd = 0	# current data cursor
	for i = 1 : nlayers
		sumd += prod(dims[i])+dims[i][1]
		offs[i] = sumd
	end
	offs
end