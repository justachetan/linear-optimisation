set Plants;
set Warehouses;


param costs{i in Plants, j in Warehouses};

param supply{i in Plants};
param demand{j in Warehouses};

var x{i in Plants, j in Warehouses};


minimize Expenditure: sum{i in Plants, j in Warehouses} costs[i, j] * x[i, j];
subject to Production{i in Plants}: sum{j in Warehouses} x[i, j] = supply[i];
subject to Demand{j in Warehouses}: sum{i in Plants} x[i, j] = demand[j];
subject to Constraint{i in Plants, j in Warehouses}: x[i, j] >= 0;


