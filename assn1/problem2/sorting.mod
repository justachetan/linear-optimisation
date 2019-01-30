# Problem 2
# Author: Aditya Chetan
# Roll No.: 2016217

param slength; # Length of sequence

param seq{i in 1 .. slength}; # Actual sequence


var costs{i in 1 .. slength}; # Decision variable. The smallest k indices get set to 1 and rest 0

param k; # To get sum of smallest k numbers


minimize total: sum{i in 1 .. slength} seq[i] * costs[i]; # Objective is to minimise total cost of selected indices
subject to cost_constraint: sum{i in 1 .. slength} costs[i] = k; # But total cost cannot be more than required number of indices
subject to cost_bound{i in 1 .. slength}: 0 <= costs[i] <= 1; # Costs must also be between 0 and 1











