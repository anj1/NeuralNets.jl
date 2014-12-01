# Types and function definitions for multi-layer perceptrons

type NNLayer{T}
    w::AbstractMatrix{T}
    b::AbstractVector{T}
    a::Function
    ad::Function

    # sparsity
    sparse::Bool
    sparsecoef::T
	sparsity::T
end

# default with no sparsity
NNLayer{T}(w::AbstractMatrix{T},b::AbstractVector{T},a::Function,ad::Function) =
	NNLayer(w,b,a,ad,false,0.0,0.0)

type MLP 
    net::Vector{NNLayer}
    report::TrainReport
end

# In all operations between two NNLayers, the activations functions are taken from the first NNLayer
copy(l::NNLayer) = NNLayer(l.w,l.b,l.a,l.ad,l.sparse,l.sparsecoef,l.sparsity)
applylayer(l::NNLayer, x::Array) = l.w*x .+ l.b
.*(c::Number, l::NNLayer)  = begin l2=copy(l); l2.w=l.w*c;    l2.b=l.b*c;    l2 end
.*(l::NNLayer, m::NNLayer) = begin l2=copy(l); l2.w=l.w.*m.w; l2.b=l.b.*m.b; l2 end
*(l::NNLayer, m::NNLayer)  = begin l2=copy(l); l2.w=l.w.*m.w; l2.b=l.b.*m.b; l2 end
/(l::NNLayer, m::NNLayer)  = begin l2=copy(l); l2.w=l.w./m.w; l2.b=l.b./m.b; l2 end
^(l::NNLayer, c::Float64)  = begin l2=copy(l); l2.w=l.w.^c;   l2.b=l.b.^c;   l2 end
-(l::NNLayer, m::NNLayer)  = begin l2=copy(l); l2.w=l.w.-m.w; l2.b=l.b.-m.b; l2 end
.-(l::NNLayer, c::Number)  = begin l2=copy(l); l2.w=l.w.-c;   l2.b=l.b.-c;   l2 end
+(l::NNLayer, m::NNLayer)  = begin l2=copy(l); l2.w=l.w+m.w;  l2.b=l.b+m.b;  l2 end
.+(l::NNLayer, c::Number)  = begin l2=copy(l); l2.w=l.w.+c;   l2.b=l.b.+c;   l2 end
.+(c::Number, l::NNLayer)  = l .+ c

# generate an MLP autoencoder from existing layer
function autenc(::Type{MLP}, l::NNLayer, act, actd)
	dims = [size(l.w), size(l.w')]
	net = [l, NNLayer(l.w', zeros(size(l.w,2)), act, actd)]
	mlp = MLP(net)
end

function MLP(genf::Function, layer_sizes::Vector{Int}, act::Vector{Function}, actd::Vector{Function})
	# some initializations
	nlayers = length(layer_sizes) - 1
	dims = [(layer_sizes[i+1],layer_sizes[i]) for i in 1:nlayers]

	net = [NNLayer(rand(dims[i]...),rand(dims[i][1]),act[i],actd[i]) for i=1:nlayers]

	MLP(net, TrainReport())
end