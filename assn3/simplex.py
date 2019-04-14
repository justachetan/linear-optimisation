#!/usr/bin/env python
# coding: utf-8

# In[100]:


import random
import itertools
import operator
import numpy as np 
import scipy as sp
import scipy.linalg as spla
np.set_printoptions(precision=4, linewidth=np.nan)


# In[241]:


def nf2DualLP(filename):
    
    # assumes that first row is source and last is sink like in the question
    # edges will be numbered as if they are being read row-by-row left to right
    # vertices will be numbered by row
    
    nf = np.loadtxt(filename)
    for i in range(nf.shape[0]):
        nf[i, i] = 0
    numedges = np.count_nonzero(nf)
    numvertices = nf.shape[0] - 2     # non terminal vertices
    numslacks = numedges
    slack_counter = 0
    edge_counter = 0
    dual_constraints = np.zeros((numedges, numedges + numvertices + numslacks + 1))
    obj = np.zeros(2 * numedges + numvertices)
    
    for i in range(numvertices + 2):
        for j in range(numvertices + 2):
            if nf[i, j] != 0:

                obj[edge_counter] = nf[i, j]
                
                if i == 0:
                    dual_constraints[edge_counter, edge_counter] = 1
                    dual_constraints[edge_counter, numedges + j - 1] = 1
                    dual_constraints[edge_counter, numedges + numvertices + slack_counter] = -1
                    dual_constraints[edge_counter, -1] = 1
                    edge_counter+=1
                    slack_counter+=1
                
                elif j == numvertices + 1:
                    dual_constraints[edge_counter, edge_counter] = 1
                    dual_constraints[edge_counter, numedges + i - 1] = -1
                    dual_constraints[edge_counter, numedges + numvertices + slack_counter] = -1
                    dual_constraints[edge_counter, -1] = 0
                    edge_counter+=1
                    slack_counter+=1
                    
                else:
                    dual_constraints[edge_counter, edge_counter] = 1
                    dual_constraints[edge_counter, numedges + i - 1] = -1
                    dual_constraints[edge_counter, numedges + j - 1] = 1
                    dual_constraints[edge_counter, numedges + numvertices + slack_counter] = -1
                    edge_counter+=1
                    slack_counter+=1
    
    sign_constraints = np.block([
        [np.eye(numedges), np.zeros((numedges, numvertices + numslacks + 1))],
        [np.zeros((numslacks, numedges + numvertices)), np.eye(numslacks), np.ones(numedges).reshape(1, numedges).T]
    ])
    
    LPMatrix = np.vstack((dual_constraints, sign_constraints))
    
    
    return LPMatrix, obj
                    
def nf2PrimalLP(filename):
    nf = np.loadtxt(filename)
    for i in range(nf.shape[0]):
        nf[i, i] = 0
    numedges = np.count_nonzero(nf)
    numvertices = nf.shape[0] - 2
    numslacks = numedges
    slack_counter = 0
    edge_counter = 0
    
    primal_constraints = np.zeros((numedges + numvertices + 2, numedges + numslacks + 1))
    
    obj = np.zeros(numedges + numslacks)
    
    for i in range(numvertices + 2):
        for j in range(numvertices + 2):
            if nf[i, j] != 0:
                if i == 0:
                    obj[edge_counter] = -1
                    primal_constraints[edge_counter, edge_counter] = 1
                    primal_constraints[edge_counter, numedges + slack_counter] = 1
                    primal_constraints[edge_counter, -1] = nf[i, j]
                    primal_constraints[numedges + j, edge_counter] = 1 
                    edge_counter+=1
                    slack_counter+=1
                elif j == numvertices + 1:
                    primal_constraints[edge_counter, edge_counter] = 1
                    primal_constraints[edge_counter, numedges + slack_counter] = 1
                    primal_constraints[edge_counter, -1] = nf[i, j]
                    primal_constraints[numedges + i, edge_counter] = -1
                    edge_counter+=1
                    slack_counter+=1
                else:
                    primal_constraints[edge_counter, edge_counter] = 1
                    primal_constraints[edge_counter, numedges + slack_counter] = 1
                    primal_constraints[edge_counter, -1] = nf[i, j]
                    primal_constraints[numedges + i, edge_counter] = -1
                    primal_constraints[numedges + j, edge_counter] = 1
                    edge_counter+=1
                    slack_counter+=1
    
    sign_constraints = np.hstack((np.eye(2*numedges), np.zeros(2*numedges).reshape(1, 2*numedges).T))
    
    LPMatrix = np.vstack((primal_constraints, sign_constraints))
    
    return LPMatrix, obj


# In[232]:


constraints, obj = nf2DualLP("nf1.dat")


# In[234]:


splex = Simplex(10, 20, obj, constraints, 1000)
sol = splex.run_simplex2()
print(str(sol))


# In[236]:


constraints2, obj2 = nf2DualLP("nf2.dat")
splex = Simplex(29, 68, obj2, constraints2, 1000)
sol = splex.run_simplex2()
print(str(sol))


# In[245]:


constraints3, obj3 = nf2PrimalLP("nf1.dat")
splex = Simplex(14, 20, obj3, constraints3, 1000)
sol = splex.run_simplex2()
print(str(sol))


# In[243]:


constraints4, obj4 = nf2PrimalLP("nf2.dat")
splex = Simplex(39, 58, obj4, constraints4, 1000)
sol = splex.run_simplex2()
print(str(sol))


# In[250]:


obj4


# In[101]:


class LPSolution(object):
    
    def __init__(self, num_vars=0, var_vals=list(), obj=0):
        
        self.num_vars = num_vars
        self.obj = obj
        self.var_vals = var_vals
        
    def __str__(self):
        
        sol = ""
        sol += "\tSolution to the LP is as follows:\n\n"
        sol += "\t\tc'x*\t=\t" + str(self.obj) + "\n\n"
        for i in range(self.num_vars):
            if i in self.var_vals:
                sol += "\t\tx_" + str(i + 1) + "\t=\t" + str(self.var_vals[i]) + "\n"
            else:
                sol += "\t\tx_" + str(i + 1) + "\t=\t" + str(0.0) + "\n"
        return sol


# In[189]:


class Simplex(object):
    # num_eq_constraints : no. of equality constraints
    def __init__(self, num_eq_constraints, num_vars, objective, constraints=None, max_iter=100):

        self.num_eq_constraints = num_eq_constraints
        self.num_vars = num_vars
        self.c = objective
        if constraints is not None:
            self.constraints = constraints
            print("self.num_eq_constraints", self.num_eq_constraints)
            self.A = self.constraints[:self.num_eq_constraints, :-1]
            self.b = self.constraints[:self.num_eq_constraints, -1]
            

        else:
            self.A = None
            self.b = None
            self.basic_columns = None
        self.solution = None
        self.tableau = None
        self.max_iter = max_iter
        
    
    def set_constraints(self, const):
        self.constraints = const
    
    def get_num_constraints(self):
        return self.num_constraints
    
    def get_num_vars(self):
        return self.num_vars
    
    def get_constraints(self):
        return self.constraints
    
    
    def fetch_constraints(self, filename):
        self.constraints = np.loadtxt(filename)
        self.b = self.constraints[:self.num_eq_constraints, -1]
        self.constraints  = self.constraints[:, :-1]
        self.A = self.constraints[:self.num_eq_constraints, :]
#         self.get_nondg_basic_columns()
    
    def get_sol(self):
        return self.sol
    
    def get_nondg_basic_columns(self):
        basic_columns = None
        for i in itertools.combinations(range(self.A.shape[1]), self.A.shape[0]):
            B_poss = self.A[:, list(i)]                       # Possible B
#             print(B_poss)
            pivots = np.linalg.matrix_rank(B_poss)
            if pivots == B_poss.shape[1]:
                if basic_columns is None:
                    basic_columns = list()
                x_b = np.linalg.solve(B_poss, self.b)
                if (x_b > 0).all():
                    basic_columns.append(list(i))
                    if self.A.shape[0] > 50 and len(basic_columns) == 1:
                        break
                else:
                    continue
                

        if basic_columns is None:
            raise RuntimeError("No initial non-degenrate BFS detected! Terminating algorithm...")
#         basic_columns = random.choice(basic_columns)
        self.basic_columns = basic_columns
    
    def get_first_basic_columns(self):
        basic_columns = random.choice(self.basic_columns)
        return basic_columns
    
    def get_first_B(self, basic_cols):
        return self.A[:, basic_cols]
    
    def run_phase1(self):
        
        c = np.hstack([np.zeros(self.A.shape[1]), np.ones(self.A.shape[0])])
        ph1_tableau = None
        A = np.hstack([self.A.copy(), np.eye(self.A.shape[0])])
        basic_columns = (self.A.shape[1]) + np.arange(self.A.shape[0])
        B = A[:, basic_columns]
        b = self.b.copy()
        c_B = c[basic_columns]
        zeroth_row = -1 * np.hstack([np.dot(c_B, self.A), np.zeros(self.A.shape[0])])
        zeroth_col = b
        zeroth_element = -1 * np.sum(b)
        
        rest = A.copy()
        
        ph1_tableau = np.block([
            [zeroth_element, zeroth_row],
            [zeroth_col.reshape(1, zeroth_col.shape[0]).T, rest]
        ])
        
        iters = 0
        
        while (ph1_tableau[0, 1:] < 0).any():
            print("---------------------This is iteration", iters+1)
            j = np.where(ph1_tableau[0, 1:] < 0)[0][0]     # incoming basis direction
            theta = [i for i in range(1, ph1_tableau.shape[0]) if ph1_tableau[i, j + 1] > 0][0]
            for i in range(1, ph1_tableau.shape[0]): 
                if ph1_tableau[i, j + 1] > 0 and ph1_tableau[i, 0] / ph1_tableau[i, j + 1] >= 0:
                    if ph1_tableau[i, 0] / ph1_tableau[i, j + 1]  < ph1_tableau[theta, 0] / ph1_tableau[theta, j + 1]:
                        theta = i
#                         elif tableau[i, 0] / tableau[i, j + 1]  == tableau[theta, 0] / tableau[theta, j + 1]:
#                             if i in basic_columns:
#                                 continue
#                             else:
#                                 theta = i
            print(basic_columns)
            basic_columns[theta - 1] = j
#                 basic_columns.insert(theta - 1, j)
            pivot_row = theta          # index of direction which will exit the basis matrix
            pivot_col = j + 1          # direction which will enter the basis
            print(pivot_row, pivot_col)
            ph1_tableau[pivot_row, :] = ph1_tableau[pivot_row, :] / ph1_tableau[pivot_row, pivot_col]
            print("before", ph1_tableau[0, 0])
            for i in range(ph1_tableau.shape[0]):
                if i == pivot_row:
                    continue        
                ph1_tableau[i, :] = ph1_tableau[i, :] - (ph1_tableau[i, pivot_col] / ph1_tableau[pivot_row, pivot_col]) * ph1_tableau[pivot_row, :]
            print("after", ph1_tableau[0, 0])
            print(basic_columns, ph1_tableau[pivot_row, pivot_col])
            print(ph1_tableau)
            iters+=1
            if iters == self.max_iter:
                raise RuntimeError("Cycling encountered! Method could not converge in max_iter = %d iterations. Terminating..." %(self.max_iter))

        if ph1_tableau[0, 0] > 0:
            raise RuntimeError("Given LP is infeasible!")
            
        elif ph1_tableau[0, 0] == 0:
            
            if (basic_columns < self.A.shape[1]).all():
                print("ph1_tableau_shape", ph1_tableau.shape)
                return ph1_tableau[1:, :self.A.shape[1]+1], basic_columns
            
            else:
                
                while True:
                    av_inbasis_at = np.where(basic_columns >= self.A.shape[1])[0].tolist()
                    # ------------------------
                    
                    if (ph1_tableau[av_inbasis_at[0] + 1, 1:self.A.shape[1] + 1] == 0).all():
                        print("-------------------------hi-------------------------------")
                        ph1_tableau = np.delete(ph1_tableau, (av_inbasis_at[0] + 1), axis=0)
                        self.A = np.delete(self.A, av_inbasis_at[0], axis=0)
                        self.b = np.delete(self.b, av_inbasis_at[0])
                        basic_columns = np.delete(basic_columns, av_inbasis_at[0])
                        print(basic_columns, self.A.shape, self.b)
    #                     return ph1_tableau[1:, :self.A.shape[1]], basic_columns

                    else:
    #                     while(not (basic_columns < self.A.shape[1]).all()):
#                         av_inbasis_at = np.where(basic_columns >= self.A.shape[1])[0]
                        pivot_row = av_inbasis_at[0] + 1
                        pivot_col = np.where(ph1_tableau[pivot_row, 1:] != 0) + 1
                        ph1_tableau[pivot_row, :] = ph1_tableau[pivot_row, :] / ph1_tableau[pivot_row, pivot_col]
                        for i in range(ph1_tableau.shape[0]):
                            if i == pivot_row:
                                continue        
                            ph1_tableau[i, :] = ph1_tableau[i, :] - (ph1_tableau[i, pivot_col] / ph1_tableau[pivot_row, pivot_col]) * ph1_tableau[pivot_row, :]
                        basic_columns[av_inbasis_at[0]] = pivot_col - 1
                    av_inbasis_at = np.where(basic_columns >= self.A.shape[1])[0].tolist()
                    if len(av_inbasis_at) == 0:
                        break
    #                     return ph1_tableau[1:, :self.A.shape[1]], basic_columns

        print("ph1_tableau_shape", ph1_tableau.shape)
        return ph1_tableau[1:, :(self.A.shape[1] + 1)], basic_columns
    
    def run_phase2(self, tableau, basic_columns):
        self.tableau = tableau.copy()
        print(tableau)
        iters = 0
        print(basic_columns)
        while (tableau[0, 1:] < 0).any():
            j = np.where(tableau[0, 1:] < 0)[0][0]     # incoming basis direction
            theta = [i for i in range(1, tableau.shape[0]) if tableau[i, j + 1] > 0][0]
            for i in range(1, tableau.shape[0]): 
                if tableau[i, j + 1] > 0 and tableau[i, 0] / tableau[i, j + 1] >= 0:
                    if tableau[i, 0] / tableau[i, j + 1]  < tableau[theta, 0] / tableau[theta, j + 1]:
                        theta = i
#                         elif tableau[i, 0] / tableau[i, j + 1]  == tableau[theta, 0] / tableau[theta, j + 1]:
#                             if i in basic_columns:
#                                 continue
#                             else:
#                                 theta = i
            print(basic_columns)
            basic_columns[theta - 1] = j
#                 basic_columns.insert(theta - 1, j)
            pivot_row = theta          # index of direction which will exit the basis matrix
            pivot_col = j + 1          # direction which will enter the basis
            print(pivot_row, pivot_col)
            tableau[pivot_row, :] = tableau[pivot_row, :] / tableau[pivot_row, pivot_col]
            print("before",tableau[0, 0])
            for i in range(tableau.shape[0]):
                if i == pivot_row:
                    continue        
                tableau[i, :] = tableau[i, :] - (tableau[i, pivot_col] / tableau[pivot_row, pivot_col]) * tableau[pivot_row, :]
            print("after",tableau[0, 0])
            print(basic_columns, tableau[pivot_row, pivot_col])
            print(tableau)
            iters+=1
            if iters == self.max_iter:
                raise RuntimeError("\n\nMethod could not converge in max_iter = %d iterations. Terminating method...!\n\n" %(self.max_iter))
                

        self.solution = LPSolution(self.num_vars, {basic_columns[i]: tableau[1:, 0][i] for i in range(len(basic_columns))}, tableau[0, 0])

        return self.solution
        

    def run_simplex2(self):
        lower_half_tableau, initial_basis = self.run_phase1()
        print(lower_half_tableau.shape, self.A.shape)
        b = self.b.copy()
        B = self.get_first_B(initial_basis)
        c_B = self.c[initial_basis]
        zeroth_element = -1 * np.dot(c_B, np.linalg.solve(B, b))
        zeroth_row = self.c - np.dot(c_B, np.dot(np.linalg.inv(B), self.A))
        print("---------------------------")
        print(zeroth_element, zeroth_row, lower_half_tableau)
        print(np.hstack((zeroth_element, zeroth_row)).shape, lower_half_tableau.shape)
        tableau = np.vstack((np.hstack((zeroth_element, zeroth_row)), lower_half_tableau))
#         tableau = np.block([
#             [zeroth_element, zeroth_row],
#             [lower_half_tableau]
#         ])
        
        self.solution = self.run_phase2(tableau, initial_basis)
        
        return self.solution
        
        
        
    
    def form_tableau(self):
        basic_columns = self.get_first_basic_columns()
        b = self.b
        B = self.get_first_B(basic_columns)
        c_B = self.c[basic_columns]
        zeroth_element = -1 * np.dot(c_B, np.linalg.solve(B, b))
        zeroth_col = np.linalg.solve(B, b)
#         print(B)
        zeroth_row = self.c - np.dot(c_B, np.dot(np.linalg.inv(B), self.A))
        rest = np.dot(np.linalg.inv(B), self.A)
        tableau = np.block([
            [zeroth_element, zeroth_row],
            [zeroth_col.reshape(1, zeroth_col.shape[0]).T, rest]
        ])
        return tableau, basic_columns
    
#     def display_tableau(self, basic_cols, tableau):
#         print("\t\t")
#         for i in tableau:
#             for 
    
    def run_simplex(self):
        while True:
            tableau, basic_columns = self.form_tableau()
            self.tableau = tableau.copy()
            print(tableau)
            iters = 0
            flag = 0                                        # indicates if we have the right initial BFS or not
            while (tableau[0, 1:] < 0).any():
                j = np.where(tableau[0, 1:] < 0)[0][0]     # incoming basis direction
                theta = [i for i in range(1, tableau.shape[0]) if tableau[i, j + 1] > 0][0]
                for i in range(1, tableau.shape[0]): 
                    if tableau[i, j + 1] > 0 and tableau[i, 0] / tableau[i, j + 1] >= 0:
                        if tableau[i, 0] / tableau[i, j + 1]  < tableau[theta, 0] / tableau[theta, j + 1]:
                            theta = i
#                         elif tableau[i, 0] / tableau[i, j + 1]  == tableau[theta, 0] / tableau[theta, j + 1]:
#                             if i in basic_columns:
#                                 continue
#                             else:
#                                 theta = i
                print(basic_columns)
                basic_columns[theta - 1] = j
#                 basic_columns.insert(theta - 1, j)
                pivot_row = theta          # index of direction which will exit the basis matrix
                pivot_col = j + 1          # direction which will enter the basis
                print(pivot_row, pivot_col)
                tableau[pivot_row, :] = tableau[pivot_row, :] / tableau[pivot_row, pivot_col]
                print("before",tableau[0, 0])
                for i in range(tableau.shape[0]):
                    if i == pivot_row:
                        continue        
                    tableau[i, :] = tableau[i, :] - (tableau[i, pivot_col] / tableau[pivot_row, pivot_col]) * tableau[pivot_row, :]
                print("after",tableau[0, 0])
                print(basic_columns, tableau[pivot_row, pivot_col])
                print(tableau)
                iters+=1
                if iters == self.max_iter:
                    print("\n\nCycling encountered! Method could not converge in max_iter = %d iterations. Restarting with new BFS!\n\n" %(self.max_iter))
                    self.basic_columns.pop(self.basic_columns.index(basic_columns))
                    flag = 1                             # not the right initial BFS!
                    break
                    
            if flag == 0:
                self.solution = LPSolution(self.num_vars, {basic_columns[i]: tableau[1:, 0][i] for i in range(len(basic_columns))}, tableau[0, 0])
        
                return self.solution    


# In[246]:


simplex = Simplex(39, 58, np.array([-1.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]))


# In[247]:


simplex.fetch_constraints("flow2.dat")


# In[248]:


sol = simplex.run_simplex2()


# In[249]:


print(str(simplex.solution))


# In[73]:


simplex.A.shape[1] + 1


# In[391]:


simplex2 = Simplex(5, 9, np.array([0.0,0,0,-2,0,0,0,0,-1]))
simplex2.fetch_constraints("constraints3.dat")
sol2 = simplex2.run_simplex()
print(str(sol2))


# In[136]:


simplex3 = Simplex(4, 4, np.array([1, 1, 1, 0.0]))
simplex3.fetch_constraints("constraints4.dat")
sol3 = simplex3.run_simplex2()
print(str(sol3))


# In[274]:


B = simplex.A[:,[0, 1, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]]


# In[275]:


X = np.dot(np.linalg.inv(B), A)


# In[277]:


c_B = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[[0, 1, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]]
Y = np.dot(c_B, X)
c = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
c - Y


# In[215]:


C = np.loadtxt("flow1.dat")


# In[216]:


A = C[:14]


# In[217]:


b = A[:, -1]
A = A[:, :-1]


# In[218]:


A, b, A.shape


# In[219]:


basic_columns = None
for i in itertools.combinations(range(A.shape[1]), A.shape[0]):
    B_poss = A[:, list(i)]                       # Possible B
    pivots = np.linalg.matrix_rank(B_poss)
    if pivots == B_poss.shape[1]:
        if basic_columns is None:
            basic_columns = list()
        x_b = np.linalg.solve(B_poss, b)
        if (x_b > 0).all():
            basic_columns.append(list(i))
        else:
            continue

if basic_columns is None:
    raise RuntimeError("No initial non-degenrate BFS detected! Terminating algorithm")
basic_columns = random.choice(basic_columns)


# In[220]:


basic_columns = list(basic_columns)
basic_columns


# In[221]:


B = A[:, basic_columns]


# In[222]:


c = np.loadtxt("flow1_objective.dat")
c_b = c[basic_columns]
zeroth_element = -1 * np.dot(c_b, np.linalg.solve(B, b))
zeroth_col = np.linalg.solve(B, b)
zeroth_row = c - np.dot(c_b, np.matmul(np.linalg.inv(B), A))
rest = np.dot(np.linalg.inv(B), A)


# In[223]:


rest.shape, zeroth_element.shape, zeroth_col.shape, zeroth_row.shape


# In[224]:


c, c_b


# In[225]:


zeroth_row, zeroth_element


# In[226]:


tableau = np.block([
    [zeroth_element, zeroth_row],
    [zeroth_col.reshape(1, 14).T, rest]
])


# In[227]:


tableau


# In[228]:


basic_columns


# In[229]:


while (tableau[0, 1:] < 0).any():
    print(basic_columns)
    j = np.where(tableau[0, 1:] < 0)[0][0]     # incoming basis direction
    theta = 1
    for i in range(1, tableau.shape[0]): 
        if tableau[i, j + 1] > 0 and tableau[i, 0] / tableau[i, j + 1] >= 0:
            if tableau[i, 0] / tableau[i, j + 1]  < tableau[theta, 0] / tableau[theta, j + 1]:
                theta = i
    basic_columns.pop(theta - 1)
    basic_columns.insert(theta - 1, j)
    pivot_row = theta          # index of direction which will exit the basis matrix
    pivot_col = j + 1          # direction which will enter the basis
    print(pivot_row, pivot_col)
    tableau[pivot_row, :] = tableau[pivot_row, :] / tableau[pivot_row, pivot_col]
    print("before",tableau[0, 0])
    for i in range(tableau.shape[0]):
        if i == pivot_row:
            continue        
        tableau[i, :] = tableau[i, :] - (tableau[i, pivot_col] / tableau[pivot_row, pivot_col]) * tableau[pivot_row, :]
    print("after",tableau[0, 0])
    print(basic_columns, tableau[pivot_row, pivot_col])
    print(tableau)


# In[ ]:


pivot_row, pivot_col


# In[ ]:


simplex.c


# In[ ]:


a = np.eye(10, dtype=np.int64)
b = np.eye(10, dtype=np.int64)
c = np.hstack([a, b])
d = np.array([16, 13, 10, 4, 12, 9, 14, 7, 20, 7])
e = np.hstack([c, d.reshape(d.shape[0], 1)])
for i in e: 
    for j in i: 
        print(j, end=" ")
    print("")


# In[ ]:


f = np.eye(20, dtype=np.int64)
g = np.zeros(20, dtype=np.int64).reshape((f.shape[0], 1))
h = np.hstack([f, g])
for i in h: 
    for j in i: 
        print(j, end=" ")
    print("")


# In[ ]:


a = "4"
a+="5"


# In[72]:


np.zeros(20, dtype=np.int64)


# In[313]:


a = [(1, 2), (3, 4), (-1, 4)]


# In[314]:


a


# In[317]:


a.sort(key=operator.itemgetter(0))


# In[318]:


a


# In[319]:


{i: i for i in range(10)}


# In[377]:


for i in np.hstack([np.eye(9), np.zeros(9).reshape((9, 1))]):
    for j in i:
        print(str(j), end=" ")
    print()


# In[370]:


np.zeros(9).reshape((9, 1))


# In[392]:





# In[398]:


tab = simplex.tableau.copy()
num_vars = 20


# In[399]:


s = ""
for i in range(num_vars):
    s+="x_" +str(i+1)+"\t"
s+="\n"


# In[402]:


print(s)


# In[406]:


a1 = np.hstack([np.eye(29), np.eye(29)])
a2 = np.array([11, 15, 10, 18, 4, 3, 5, 6, 3, 11, 4, 17, 6, 3, 13, 12, 4, 21, 4, 9, 4, 3, 4, 5, 4, 7, 9, 2, 15])
a3 = np.hstack([a1, a2.reshape(a2.shape[0], 1)])


# In[409]:


for i in a3:
    for j in i:
        print(j, end=",")
    print()


# In[411]:


for i in np.zeros((58, 58)):
    for j in i:
        print(j, end=" ")
    print()


# In[414]:


print(np.zeros(58).tolist())


# In[3]:


np.sum(np.array([2, 3, 4]))


# In[5]:


np.random.rand(3, 3)[:]


# In[8]:


print( not True)


# In[185]:


np.delete(np.arange(4), 3)


# In[33]:


np.hstack((1, np.array([1, 2, 3])))


# In[40]:


np.random.rand(4, 4)


# In[152]:


np.zeros((3,3))


# In[ ]:




