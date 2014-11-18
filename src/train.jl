type TrainReport
    algorithm::String
    train_parameters::Dict
    iteration::Vector{Int}
    train_error::Vector{Real}
    valid_error::Vector{Real}
    trained::Bool

    function TrainReport(algorithm::String,train_parameters::Dict)
        new(algorithm,train_parameters,Int[],Real[],Real[],false)
    end
end

function Base.show(io::IO, h::TrainReport)
    println(h.algorithm)
    for p in h.train_parameters
        println("$(p[1]): ", p[2])
    end
    if length(h.iteration) > 0
        if length(h.valid_error) < length(h.train_error) # if there's no validation set
            @printf "Iteration      Train error\n"
            @printf "---------   --------------\n"
            for i = 1:length(h.iteration)
                @printf "%9i   %14e\n" h.iteration[i] h.train_error[i]
            end
            @printf "--------------------------\n"
        else
            @printf "Iteration      Train error     Valid. error\n"
            @printf "---------   --------------   --------------\n"
            for i = 1:length(h.iteration)
                @printf "%9i   %14e   %14e\n" h.iteration[i] h.train_error[i] h.valid_error[i]
            end
            @printf "-------------------------------------------\n"
        end
    end
end

# basically just echo back to the user what dumbass settings they’ve picked
function display_training_header!(nnet::MLP, h::TrainReport)
    info("now training with the following parameters")
    for p in h.train_parameters
        print_with_color(:blue,"$(p[1]): $(p[2])\n")
    end
    @printf "-------------------------------------------\n"
end

# store and show trace of loss for diagnostic purposes
function diagnostic_trace!(h::TrainReport,
                           i::Int,
                           train_error::Real,
                           valid_error::Real,
                           valid::Bool,
                           show_trace::Bool,
                           in_place::Bool,
                           converged::Bool)
    converged && (h.trained = converged)
    if valid # if a validation set is present
        push!(h.iteration, i)
        push!(h.train_error, train_error)
        push!(h.valid_error, valid_error)
    else
        push!(h.iteration, i)
        push!(h.train_error, train_error)
    end
    show_trace && display_status!(h; in_place=in_place)
end

# let the dumbass user know how their stupid settings fare
function display_training_footer!(h::TrainReport, in_place::Bool)
    in_place && print("\n")
    i = h.iteration[end]
    if h.trained
        info("training converged after around $i iterations")
    else
        warn("training failed to converge after around $i iterations")
    end
end

# display current training progress
function display_status!(h::TrainReport; in_place=true)
    if length(h.valid_error) < length(h.train_error) # if there's no validation set
        if length(h.iteration) == 1
            @printf "i: %9i   train error: %14e" h.iteration[end] h.train_error[end]
        else
            @printf "i: %9i" h.iteration[end]

            train_color = h.train_error[end] < h.train_error[end-1] ? :green : :red
            train_string = @sprintf "%14e" h.train_error[end]
            print("   train error: ")
            print_with_color(train_color, train_string)
        end
    else
        if length(h.iteration) == 1
            @printf "i: %9i   train error: %14e   valid. error: %14e\r" h.iteration[end] h.train_error[end] h.valid_error[end]
        else
            @printf "i: %9i" h.iteration[end]

            train_color = h.train_error[end] < h.train_error[end-1] ? :green : :red
            train_string = @sprintf "%14e" h.train_error[end]
            print("   train error: ")
            print_with_color(train_color, train_string)

            valid_color = h.valid_error[end] < h.valid_error[end-1] ? :green : :red
            valid_string = @sprintf "%14e" h.valid_error[end]
            print("   valid. error: ")
            print_with_color(valid_color, valid_string)
        end
    end
    in_place ? print("\r") : print("\n")
end