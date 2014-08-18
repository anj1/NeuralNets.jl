# layer-wise pretraining for deep autoencoders
# net is the layers of the net
# x: training data
# xn: noisy training data
function pretrain(net::Vector{T}, x, xn, act, actd, trainfunc, trainargs...)
	for i = 1 : length(nn.net)-1
		# build this network layer (TODO: make a bit more generic)
		l = nn.net[i]
		netp = [l, T(l.w', zeros(size(x,1)), act, actd)]

		# train this network layer
		# TODO: needs to call with entire structure, not net
		trainfunc(netp, x, xn, trainargs...)

		# propagate inputs & noisy inputs
		x  = prop([l], x)
		xn = prop([l], xn)
	end
end
