module PostgreSQL
    export  Postgres,
            executemany,
            escapeliteral,
            execute,
            prepare

    include("libpq.jl")
    using .libpq
    using DBI
    using DataFrames
    using DataArrays
    include("types.jl")
    include("dba.jl")

end
