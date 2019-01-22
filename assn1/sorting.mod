param slength; # Length of sequence

param seq{i in 1 .. slength}; # Actual sequence


var costs{i in 1 .. slength};

param k; # To get sum of smallest k numbers


minimize total: sum{i in 1 .. slength} seq[i] * costs[i];
subject to cost_constraint: sum{i in 1 .. slength} costs[i] = k;
subject to cost_bound{i in 1 .. slength}: 0 <= costs[i] <= 1;











