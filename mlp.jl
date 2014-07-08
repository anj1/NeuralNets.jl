# Types and function definitions for multi-layer perceptrons

# Top level MLP type for predicting things and diagnosing convergence
type MLP{T}
    net::Vector{T}
    hidden::Vector      # topology of hidden nodes
    i::Int              # iterations for convergence
    c::Real             # convergence criterion
    η::Real             # learning rate
    # more to come...
end

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
*(l::NNLayer, x::Array) = l.w*x .+ l.b
.*(c::Number, l::NNLayer) = NNLayer(c*l.w, c*l.b, l.a, l.ad)            
-(l::NNLayer, m::NNLayer) = NNLayer(l.w - m.w, l.b - m.b, l.a, l.ad)    
-(l::NNLayer, c::Number) = NNLayer(l.w .- c, l.b .- c, l.a, l.ad)       
+(l::NNLayer, m::NNLayer) = NNLayer(l.w + m.w, l.b + m.b, l.a, l.ad)  
+(l::NNLayer, c::Number) = NNLayer(l.w .+ c, l.b .+ c, l.a, l.ad)    

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

# Train a MLP using stochastic gradient decent with momentum.
# l:        array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# eval:     how often we evaluate the loss function (20)
# verbose:  train with printed feedback about the error function (true)
function gdmtrain{T}(net::Vector{T},x,t,η::Real,c::Real,m::Real=0; eval::Int=20, verbose::Bool=true)
    i = e_old = Δ_old = 0
    e_new = loss(prop(net,x),t)
    in_dim,n = size(x)
    converged::Bool = false
    while !converged
        i += 1
        ∇,δ = backprop(net,x,t)
        Δ_new = η*∇ + m*Δ_old  # calculatew Δ weights
        net = net - Δ_new      # update weights                       
        Δ_old = Δ_new           
        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(net,x),t)
            if verbose == true
                println("i: $i\t Loss=$(round(e_new,6))\t ΔLoss=$(round((e_new - e_old),6))\t Avg. Loss=$(round((e_new/n),6))")
            end
        end
        # check for convergence            
        abs(e_new - e_old) < c ? converged = true : nothing
    end
    println("Training converged in less than $i iterations with average error: $(round((e_new/n),4)).")
    println("* learning rate η = $η")
    println("* momentum coefficient m = $m")
    println("* convergence criterion c = $c")
    return net
end