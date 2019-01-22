param num_days;
param tenure;

param Demands{i in 1 .. num_days};

var new_hires{i in 1 .. num_days};

minimize Workers: sum{i in 1 .. num_days} new_hires[i];
subject to TenureConst{i in 1 .. num_days}: sum{j in 1 .. tenure} new_hires[(i - j + num_days) mod num_days + 1] >= Demands[i];
subject to NHConst{i in 1 .. num_days}: new_hires[i] >= 0;















