type NNLayer{T}
    w::AbstractMatrix{T}
    b::AbstractVector{T}
    a::Function
    ad::Function
end

# In all operations between two NNLayers, the activations functions are taken from the first NNLayer
*(l::NNLayer, x::Array) = l.w*x .+ l.b
*(l::NNLayer, x::Array) = l.w*x .+ l.b
.*(c::Number, l::NNLayer) = NNLayer(c*l.w, c*l.b, l.a, l.ad)            
-(l::NNLayer, m::NNLayer) = NNLayer(l.w - m.w, l.b - m.b, l.a, l.ad)    
-(l::NNLayer, c::Number) = NNLayer(l.w .- c, l.b .- c, l.a, l.ad)       
+(l::NNLayer, m::NNLayer) = NNLayer(l.w + m.w, l.b + m.b, l.a, l.ad)  
+(l::NNLayer, c::Number) = NNLayer(l.w .+ c, l.b .+ c, l.a, l.ad)  

function Base.show(io::IO, l::NNLayer)
    print(io, summary(l),":\n")
    print(io, "activation functions:\n")
    print(io, l.a,", ",l.ad,"\n")
    print(io, "node weights:\n",l.w,"\n")
    print(io, "bias weights:\n",l.b)
end  

type MLP
    net::Vector{NNLayer}
    dims::Vector{(Int,Int)}  # topology of net
    buf::AbstractVector      # in-place data store
    offs::Vector{Int}    # indices into in-place store
    trained::Bool
end

type TrainingParams
    i::Int              # iterations for convergence
    c::Real             # convergence criterion
    η::Real             # learning rate
    m::Real             # momentum amount
    train_method        # training method
end

function Base.show(io::IO, p::TrainingParams)
    print(io, summary(p),"\n")
    print(io, "Parameters for training a neural network:","\n")
    print(io, "* maximum iterations: ", p.i,"\n")
    print(io, "* convergence criterion: ", p.c,"\n")
    print(io, "* learning rate: ", p.η,"\n")
    print(io, "* momentum coefficient: ", p.m,"\n")
    print(io, "* train method: ", p.train_method,"\n")
end
