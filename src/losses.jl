# collection of commonly-used loss functions

# y = predicted value, t = expected value

squared_loss(y, t) = 0.5 * norm(y .- t).^2 # L(y, t) = .5 || y - t ||^2 = .5 (y - t)^2
squared_lossd(y, t) = y .- t # d/dx L(y, t) = (y - t)

linear_loss(y, t) = norm(y .- t, 1) # L(y, t) = || y .- t ||_1 = sum(abs(y .- t))
linear_lossd(y, t) = sign(y .- t)

hinge_loss(y, t, m=1.0) = sum(max(0, m - (y .* t))) # max(0, m - y t)
hinge_lossd(y, t, m=1.0) = - t .* (m > t .* y) # if (m > y t) then - t, else 0

log_loss(y, t) = sum(- ((t .* log(y)) + ((1 - t) .* log(1 - y)))) # - t * log(y) - (1 - t) * log(1 - y)
log_lossd(y, t) = (t .- y) ./ ((y .- 1) .* y) # (t - y) / ((y - 1) * y)

# dictionary of commonly-used loss derivatives
lossderivs = Dict{Function, Function}([
                                   squared_loss      => squared_lossd,
                                   linear_loss       => linear_lossd,
                                   hinge_loss        => hinge_lossd,
                                   log_loss          => log_lossd
                                   ])
