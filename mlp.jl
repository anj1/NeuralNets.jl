# Types and function definitions for multi-layer perceptrons
	
type NNLayer{T}
	w::Matrix{T}
	b::Vector{T}
	a::Function
	ad::Function
end

function Base.show(io::IO, l::NNLayer)
    print(io, summary(l),"\n","activation functions:\n",l.a,", ",l.ad,"\n","node weights:\n",l.w,"\n","bias weights:\n",l.b)
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
    buf::Vector      # in-place data store
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

# TODO: infer actd from act.
# If activation function has known derivative, use that.
# Otherwise, use automatic differentiation.
function MLP(genf::Function, layer_sizes::Vector{Int}, act::Vector{Function}, actd::Vector{Function})
	# some initializations
	nlayers = length(layer_sizes) - 1
	dims = [(layer_sizes[i+1],layer_sizes[i]) for i in 1:nlayers]

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
function unflatten_net!(mlp::MLP, buf::Vector)
	mlp.buf = buf

	pbuf = pointer(buf)
	for i = 1 : length(mlp.net)
		toff = i > 1 ? mlp.offs[i-1] : 0
		tpbuf = pbuf + toff*sizeof(eltype(buf))
		tdims = mlp.dims[i]
		mlp.net[i].w = pointer_to_array(tpbuf, tdims)
		mlp.net[i].b = pointer_to_array(tpbuf + sizeof(eltype(buf))*prod(tdims), (tdims[1],))
	end
end