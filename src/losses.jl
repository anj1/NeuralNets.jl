# collection of commonly-used loss functions

squared_loss(y, t) = 0.5 * norm(y .- t).^2
linear_loss(y, t) = norm(y .- t)
hinge_loss(y, t, m=1.0) = max(0, m - norm(y .* t))
log_loss(y, t) = - (y .* log(t)) + (1 - y) .* log(1 - t)
