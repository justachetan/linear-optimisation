# Problem 3
# Author: Aditya Chetan
# Roll No.: 2016217


param num_prod; # Total types of products
param num_rml; # Total types of raw materials

param profits{j in 1 .. num_prod}; # Array to store profit on each type of product
param stock{i in 1 .. num_rml}; # Array to store stock of each type of raw material


param req{i in 1 .. num_rml, j in 1 .. num_prod}; # Array to store requirement of each type of raw material for each type of product

var prod{j in 1 .. num_prod}; # Decision variable to get the production of each type of product

maximize Gains: sum{j in 1 .. num_prod} profits[j] * prod[j]; # Objective is to maximise gains
subject to RMConst{i in 1 .. num_rml}: sum{j in 1 .. num_prod} req[i, j] * prod[j] <= stock[i]; # Total usage of raw materials should not exceed stock
subject to ProdConst{j in 1 .. num_prod}: prod[j] >= 0; # Production of each type cannot be negative







