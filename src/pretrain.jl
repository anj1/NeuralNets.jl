# layer-wise pretraining for deep autoencoders
# nn is the network
# x: training data
# xn: noisy training data
function pretrain{T}(nn::T, x, xn, act, actd, trainfunc, trainargs...)
	# for each layer in the network
	for i = 1 : length(nn.net)-1
		# build this autoencoder layer
		l = nn.net[i]
		curmlp = autenc(T, l)

		# train this network layer
		curmlp = trainfunc(curmlp, x, xn, trainargs...)

		# use parameters of first layer
		nn.net[i] = curmlp.net[1]

		# propagate inputs & noisy inputs
		x  = prop([l], x)
		xn = prop([l], xn)
	end
end
