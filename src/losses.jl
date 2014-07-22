# collection of commonly-used loss functions

# y = predicted value, t = expected value

squared_loss(y, t) = 0.5 * norm(y .- t).^2 # L(x, z) = .5 || x - z ||^2 = .5 (x - z)^2
squared_lossd(y, t) = y .- t # d/dx L(x, z) = || x - z ||

linear_loss(y, t) = norm(y .- t, 1) # L(x, z) = || y .- t ||_1 = sum(abs(y .- t))
linear_lossd(y, t) = sign(y .- t)

hinge_loss(y, t, m=1.0) = sum(max(0, m - (y .* t))) # max(0, m - x * t)
hinge_lossd(y, t, m=1.0) = -t .* (m > t .* y) # if (m > t x) then - t, else 0

log_loss(y, t) = sum(- ((t .* log(y)) + ((1 - t) .* log(1 - y)))) # - t * log(y) - (1 - t) * log(1 - y)
log_lossd(y, t) = (t .- y) ./ ((y .- 1) .* y) # t - y / (y - 1) * y

# dictionary of commonly-used loss derivatives
derivs = Dict{Function, Function}([
                                   squared_loss      => squared_lossd,
                                   linear_loss       => linear_lossd,
                                   hinge_loss        => hinge_lossd,
                                   log_loss          => log_lossd
                                   ])
