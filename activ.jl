# collection of commonly-used activation functions
logis(x) = 1 ./(1 .+ exp(-x))
logisd(x) = exp(x) ./ ((1 .+ exp(x)).^2)

relu(x) = log(1 .+ exp(x))
relud(x) = 1 ./(1 .+ exp(-x))

ident(x) = x
identd(x) = 1

tanhd(x) = sech(x).^2

# dictionary of commonly-used activation derivatives
derivs = Dict{Function, Function}([logis => logisd, 
                                   relu => relud, 
                                   ident => identd, 
                                   tanh => tanhd])