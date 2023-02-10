using Gadfly, RDatasets
gasoline = dataset("Ecdat", "Gasoline")
Gadfly.plot(gasoline, x=:Year, y=:LGasPCar, color=:Country, Geom.point, Geom.line)