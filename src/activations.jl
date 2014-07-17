# collection of commonly-used activation functions
logis(x) = 1 ./(1 .+ exp(-x))
logisd(x) = exp(x) ./ ((1 .+ exp(x)).^2)

logissafe(x) = logis(x)
logissafed(x) = logisd(min(x,400.0))

relu(x) = log(1 .+ exp(x))
relud(x) = 1 ./(1 .+ exp(-x))

ident(x) = x
identd(x) = 1

tanhd(x) = sech(x).^2

# dictionary of commonly-used activation derivatives
derivs = Dict{Function, Function}([logis     => logisd, 
                                   logissafe => logissafed,
                                   relu      => relud, 
                                   ident     => identd, 
                                   tanh      => tanhd])

# automatic differentiateion with ForwardDiff.jl
# due to limitations of ForwardDiff.jl, this function
# will only produce derivatives with Float64 methods
function autodiff(activ::Function)
    f(x) = activ(x[1])
    forwarddiff_derivative(x::Float64) = forwarddiff_gradient(f,Float64)([x])[1]
    return forwarddiff_derivative
end