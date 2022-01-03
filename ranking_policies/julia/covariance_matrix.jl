using Zygote

h(x, y) = 3x^2 + 2x + 1 + y*x - y
result = gradient(h, 3.0, 5.0)
print(result)