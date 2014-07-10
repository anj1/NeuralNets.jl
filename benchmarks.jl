# bring in definitions
require("activ.jl")
require("mlp.jl")
require("backprop.jl")
require("train.jl")

using DataFrames 
using MLBase

bicycle = readtable("./datasets/bicycle_demand.csv") # load test data

function dataprep(data)
    year    = int(map(x->x[1:4],data[:datetime]))
    month   = int(map(x->x[6:7],data[:datetime]))
    day     = int(map(x->x[9:10],data[:datetime]))
    hour    = int(map(x->x[12:13],data[:datetime]))
    X = hcat(year,month,hour,array(data[[:season,:temp,:holiday,:workingday,:weather,:atemp,:humidity,:windspeed]]))'
    T = array(data[[:count]])'

    X = convert(Array{Float64,2},X)
    T = convert(Array{Float64,2},T)
    return X,T
end

X,T = dataprep(bicycle)

trans_x = estimate(Standardize, X; center=true, scale=true)
trans_t = estimate(Standardize, T; center=true, scale=true)

transform!(trans_x,X)
transform!(trans_t,T)

function untransform(t::Standardize, x::Array)
    (x .+ t.mean .* t.scale) ./ t.scale
end

ind = size(X,1)
outd = size(T,1)

layer_sizes = [ind, 3, 3, outd]
act   = [relu,  relu,  ident]
actd  = [relud, relud, identd]

# initialize net
mlp = MLP(randn, layer_sizes, act, actd)

# training parameters
params = TrainingParams(100000, 1., 4e-6, 1., :gdmtrain)

gdm_mlp = train(mlp, params, X, T)