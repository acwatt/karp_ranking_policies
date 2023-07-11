# Test savio
import Pkg; Pkg.add("BenchmarkTools")
using BenchmarkTools
# Test parallel computing using Threads.@threads
# @benchmark sort(data) setup=(data=rand(10))

function parallel_test()
    Threads.@threads for i in 1:100
        sleep(1/10000)
    end
end
@benchmark parallel_test()