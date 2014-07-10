# Train a MLP using Levenberg-Marquardt optimisation.
# mlp.net:  array of neural network layers
# x:        input data
# t:        target data
# η:        learning rate
# m:        momentum coefficient
# c:        convergence criterion
# λ:        mixing coefficient, large λ -> gradient descent, small λ -> quadratic approximated hessian
# i:        number of accepted steps
# j:        number of attempts at finding steps
# eval:     how often we evaluate the loss function
# verbose:  train with printed feedback about the error function
# TODO need to calculate hessian approximation H
function lmtrain(mlp::MLP, p::TrainingParams, x, t; eval::Int=10, verbose::Bool=true)
    η, c, m = p.η, p.c, p.m
    i = e_old = Δ_old = 0
    λ = 100
    converged::Bool = false    
    while !converged
        i += 1          
        # Start of the update step
        H = [kron(d[i],d[i]) for i in length(d)] 
        while true 
            j += 1 
            ∇,δ = backprop(mlp.net,x,t)

            J = g(mlp.buf)
            H = J'*J
            diagH = diagm(diag(H))


            e_new = loss(prop(mlp.net - Δw_new,x),t)    # test new update
            Δe = e_new - e_old
            if Δe > 0           
                λ *= 10   # if new step -> error goes up, increase λ and try again
            else                                         
                λ /= 10   # if new step -> error goes down, decrease λ
                break
            end
            if j - i > 1e4
                println("Oh god something's fucked up here")
                println("j= $j, i= $i")
                break
            end
        end
        # End of the update calculation step            
        mlp.net = mlp.net .- Δw_new     # accept step, update weights               
        Δw_old = Δw_new
        if i % eval == 0  # recalculate loss every eval number iterations
            e_old = e_new
            e_new = loss(prop(mlp.net,x),t)
            converged = abs(e_new - e_old) < c # check if converged
        end
        if i % 100 == 0 && verbose == true
            println("i: $i\t Loss=$(round(e_new,6))\t ΔLoss=$(round((e_new - e_old),6))\t Avg. Loss=$(round((e_new/n),6))")
        end        
        i >= p.i && break # check if hit the max iterations limit 
    end
end