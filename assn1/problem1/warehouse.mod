# Problem 1
# Author: Aditya Chetan
# Roll No.: 2016217

set Plants; # Set containing the plants 
set Warehouses; # Set containing the warehouses


param costs{i in Plants, j in Warehouses}; # Cost of sending one unit of products from warehouse i to plant j

param supply{i in Plants}; # Array to store supply power of each plant
param demand{j in Warehouses}; # Array to store required demand at each warehouse

var x{i in Plants, j in Warehouses}; # Decision variable to store optimum amount sent from plant i to warehouse j


minimize Expenditure: sum{i in Plants, j in Warehouses} costs[i, j] * x[i, j]; # Objective is to minimise the total expenditure of transportation
subject to Production{i in Plants}: sum{j in Warehouses} x[i, j] = supply[i]; # But total production at a plant has to be less than its supply power
subject to Demand{j in Warehouses}: sum{i in Plants} x[i, j] = demand[j]; # Ans total inflow of products at a warehouse should be equal to its demand
subject to Constraint{i in Plants, j in Warehouses}: x[i, j] >= 0; # And the amount transported between any warehouse and plant cannot be negative


