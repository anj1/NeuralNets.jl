using NeuralNets

# xor training data
x = [
    0.0 1.0 0.0 1.0
    0.0 0.0 1.0 1.0
    ]

t = [
    0.0 1.0 1.0 0.0
    ]

# network topology
layer_sizes = [2, 3, 3, 1]
act   = [relu,  relu,  logis]

# initialize net
mlp = MultiLayerPerceptron(randn, layer_sizes, act)

mlp2 = gdmtrain(mlp, x, t; store_trace=true, in_place=false)
prop(mlp,x)

using Gadfly
function plot(mlp::MultiLayerPerceptron)
    report = mlp.report
    plot(y=report.train_error, x=1:length(report.train_error),
        Guide.xlabel("Iterations"),
        Guide.ylabel("Error"),
        Geom.point,Geom.line)
end
