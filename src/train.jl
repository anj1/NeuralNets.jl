type TrainReport
    train_error::Vector
    valid_error::Vector
    converged::Bool

    function TrainReport()
        new(Real[],Real[],false)
    end
end

# store/show trace of loss for diagnostic purposes
function diagnostic_trace!(h::TrainReport,
                           train_error::Real,
                           valid_error::Real,
                           valid::Bool,
                           show_trace::Bool)
    push!(h.train_error, train_error)
    if valid # if a validation set is present
        push!(h.valid_error, valid_error)
    end
    if show_trace
        if valid
            print("training error: $train_error\r")
        else
            print("training error: $train_error, validation error: $valid_error\r")
        end
    end
end