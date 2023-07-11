# Test savio
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
t = @benchmark parallel_test()

display(t)
println(mean(t))
