# Types and function definitions for multi-layer perceptrons

using ArrayViews

type NNLayer{T}
	w::AbstractMatrix{T}
	b::AbstractVector{T}
	a::Function
	ad::Function
end

function Base.show(io::IO, l::NNLayer)
    print(io, summary(l):,"\n")
    print(io, "activation functions:\n")
    print(io, l.a,", ",l.ad,"\n")
    print(io, "node weights:\n",l.w,"\n")
    print(io, "bias weights:\n",l.b)
end

# In all operations between two NNLayers, the activations functions are taken from the first NNLayer
*(l::NNLayer, x::Array) = l.w*x .+ l.b
*(l::NNLayer, x::Array) = l.w*x .+ l.b
.*(c::Number, l::NNLayer) = NNLayer(c*l.w, c*l.b, l.a, l.ad)            
-(l::NNLayer, m::NNLayer) = NNLayer(l.w - m.w, l.b - m.b, l.a, l.ad)    
-(l::NNLayer, c::Number) = NNLayer(l.w .- c, l.b .- c, l.a, l.ad)       
+(l::NNLayer, m::NNLayer) = NNLayer(l.w + m.w, l.b + m.b, l.a, l.ad)  
+(l::NNLayer, c::Number) = NNLayer(l.w .+ c, l.b .+ c, l.a, l.ad)    

# Top level MLP type for predicting things and diagnosing convergence
type MLP
    net::Vector{NNLayer}
    dims::Vector{(Int,Int)}  # topology of net
    buf::AbstractVector      # in-place data store
    offs::Vector{Int}    # indices into in-place store
    trained::Bool
end

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

function MLP(genf::Function, layer_sizes::Vector{Int}, act::Vector{Function})
	# some initializations
	nlayers = length(layer_sizes) - 1
	dims = [(layer_sizes[i+1],layer_sizes[i]) for i in 1:nlayers]

    # generate vector of activation derivatives
    # derivs::Dict{Function,Function} is defined in activ.jl
    actd = similar(act,Function)
    for i in 1:length(act)
        if haskey(derivs,act[i]) 
            actd[i] = derivs[act[i]]            
        else
            # hook for implementation of automatic differentiation
            throw("No derivative found for the specified activation function $(act[i]).")         
        end
    end

	# offsets into the parameter vector
	offs = calc_offsets(NNLayer, dims)

	# our single data vector
	buf = genf(offs[end])

	net = [NNLayer(Array(eltype(buf),0,0),Array(eltype(buf),0),act[i],actd[i]) for i=1:nlayers]

	mlp = MLP(net, dims, buf, offs, false)
	unflatten_net!(mlp, buf)

	mlp
end

# Given a flattened vector (buf), update the neural
# net so that each weight and bias vector points into the
# offsets provided by offs
function unflatten_net!(mlp::MLP, buf::AbstractVector)
	mlp.buf = buf

	for i = 1 : length(mlp.net)
		toff = i > 1 ? mlp.offs[i-1] : 0
		tdims = mlp.dims[i]
		lenw = prod(tdims)
		mlp.net[i].w = reshape_view(view(buf, toff+1:toff+lenw), tdims)
		toff += lenw
		mlp.net[i].b = view(buf, toff+1:toff+tdims[1])
	end
end

