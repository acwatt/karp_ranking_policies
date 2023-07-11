# Test savio


using Distributed

@everywhere begin
    project_env = joinpath(splitpath(@__DIR__)[1:end-1])
    using Pkg; Pkg.activate(project_env)
    Pkg.instantiate(); Pkg.precompile()
end

@everywhere begin
    # import Pkg; Pkg.add("BenchmarkTools")
    using BenchmarkTools
    # Test parallel computing using Threads.@threads
    # @benchmark sort(data) setup=(data=rand(10))

    println("# of Threads:$(Threads.nthreads())")

    function parallel_test()
        Threads.@threads for i in 1:100
            sleep(1/100)
        end
    end
end

function distributed_test()
    status = @showprogress pmap(1:100) do i
        try
        sleep(1/100)
        true # success
        catch e
        false # failure
        end
    end
    return nothing
end

# t = @benchmark parallel_test()
t = @benchmark distributed_test()

display(t)
println(mean(t))
