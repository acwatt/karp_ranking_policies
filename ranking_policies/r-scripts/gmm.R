# sudo apt install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev
install.packages("devtools")
library("devtools")
# install_github("slphyx/WolfinR")
library(WolfinR)
WolfinR("Plot[Sin[x],{x,0,2 Pi}]", graphics=TRUE)
