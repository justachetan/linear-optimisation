# Problem 4
# Author: Aditya Chetan
# Roll No.: 2016217

param num_days; # Total number of days for each cycle, in the given case, 7
param tenure; # Total days for which a worker works

param Demands{i in 1 .. num_days}; # Array to store demand of workers per day

var new_hires{i in 1 .. num_days}; # Decision variable to store the number of new hires each day

minimize Workers: sum{i in 1 .. num_days} new_hires[i]; # Objective is to minimize the total number of new hires
subject to TenureConst{i in 1 .. num_days}: sum{j in 1 .. tenure} new_hires[(i - j + num_days) mod num_days + 1] >= Demands[i]; # But keep in mind that the constraints of demand are satisfied
subject to NHConst{i in 1 .. num_days}: new_hires[i] >= 0; # Also number of hires on a day cannot be negative















