param num_prod; # Total types of products
param num_rml; # Total types of raw materials

param profits{j in 1 .. num_prod};
param stock{i in 1 .. num_rml};


param req{i in 1 .. num_rml, j in 1 .. num_prod};

var prod{j in 1 .. num_prod};

maximize Gains: sum{j in 1 .. num_prod} profits[j] * prod[j];
subject to RMConst{i in 1 .. num_rml}: sum{j in 1 .. num_prod} req[i, j] * prod[j] <= stock[i];
subject to ProdConst{j in 1 .. num_prod}: prod[j] >= 0;







