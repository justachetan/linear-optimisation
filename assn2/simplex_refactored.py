#!/usr/bin/env python
# coding: utf-8

# In[262]:


import random
import itertools
import operator
import numpy as np
import scipy as sp
import scipy.linalg as spla
np.set_printoptions(precision=4, linewidth=np.nan)


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
    dual_constraints = np.zeros(
        (numedges, numedges + numvertices + numslacks + 1))
    obj = np.zeros(2 * numedges + numvertices)

    for i in range(numvertices + 2):
        for j in range(numvertices + 2):
            if nf[i, j] != 0:

                obj[edge_counter] = nf[i, j]

                if i == 0:
                    dual_constraints[edge_counter, edge_counter] = 1
                    dual_constraints[edge_counter, numedges + j - 1] = 1
                    dual_constraints[edge_counter, numedges +
                                     numvertices + slack_counter] = -1
                    dual_constraints[edge_counter, -1] = 1
                    edge_counter += 1
                    slack_counter += 1

                elif j == numvertices + 1:
                    dual_constraints[edge_counter, edge_counter] = 1
                    dual_constraints[edge_counter, numedges + i - 1] = -1
                    dual_constraints[edge_counter, numedges +
                                     numvertices + slack_counter] = -1
                    dual_constraints[edge_counter, -1] = 0
                    edge_counter += 1
                    slack_counter += 1

                else:
                    dual_constraints[edge_counter, edge_counter] = 1
                    dual_constraints[edge_counter, numedges + i - 1] = -1
                    dual_constraints[edge_counter, numedges + j - 1] = 1
                    dual_constraints[edge_counter, numedges +
                                     numvertices + slack_counter] = -1
                    edge_counter += 1
                    slack_counter += 1

    sign_constraints = np.block([
        [np.eye(numedges), np.zeros((numedges, numvertices + numslacks + 1))],
        [np.zeros((numslacks, numedges + numvertices)),
         np.eye(numslacks), np.ones(numedges).reshape(1, numedges).T]
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

    primal_constraints = np.zeros(
        (numedges + numvertices + 2, numedges + numslacks + 1))

    obj = np.zeros(numedges + numslacks)

    for i in range(numvertices + 2):
        for j in range(numvertices + 2):
            if nf[i, j] != 0:
                if i == 0:
                    obj[edge_counter] = -1
                    primal_constraints[edge_counter, edge_counter] = 1
                    primal_constraints[edge_counter,
                                       numedges + slack_counter] = 1
                    primal_constraints[edge_counter, -1] = nf[i, j]
                    primal_constraints[numedges + j, edge_counter] = 1
                    edge_counter += 1
                    slack_counter += 1
                elif j == numvertices + 1:
                    primal_constraints[edge_counter, edge_counter] = 1
                    primal_constraints[edge_counter,
                                       numedges + slack_counter] = 1
                    primal_constraints[edge_counter, -1] = nf[i, j]
                    primal_constraints[numedges + i, edge_counter] = -1
                    edge_counter += 1
                    slack_counter += 1
                else:
                    primal_constraints[edge_counter, edge_counter] = 1
                    primal_constraints[edge_counter,
                                       numedges + slack_counter] = 1
                    primal_constraints[edge_counter, -1] = nf[i, j]
                    primal_constraints[numedges + i, edge_counter] = -1
                    primal_constraints[numedges + j, edge_counter] = 1
                    edge_counter += 1
                    slack_counter += 1

    sign_constraints = np.hstack(
        (np.eye(2 * numedges), np.zeros(2 * numedges).reshape(1, 2 * numedges).T))

    LPMatrix = np.vstack((primal_constraints, sign_constraints))

    return LPMatrix, obj


class LPSolution(object):

    def __init__(self, num_vars=0, var_vals=list(), obj=0):

        self.num_vars = num_vars
        self.obj = obj
        self.var_vals = var_vals

    def __str__(self):

        sol = ""
#         sol += "\tSolution to the LP is as follows:\n\n"
        sol += "optim\t:=\t" + str(self.obj) + "\n\n"
        for i in range(self.num_vars):
            if i in self.var_vals:
                sol += "x_" + str(i + 1) + "*\t:=\t" + \
                    str(self.var_vals[i]) + "\n"
            else:
                sol += "x_" + str(i + 1) + "*\t:=\t" + str(0.0) + "\n"
        return sol


class Simplex(object):
    # num_eq_constraints : no. of equality constraints

    def __init__(self, num_eq_constraints, num_vars, objective, constraints=None, max_iter=100):

        self.num_eq_constraints = num_eq_constraints
        self.num_vars = num_vars
        self.c = objective
        if constraints is not None:
            self.constraints = constraints
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
        self.constraints = self.constraints[:, :-1]
        self.A = self.constraints[:self.num_eq_constraints, :]

    def get_sol(self):
        return self.sol

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
        zeroth_row = -1 * \
            np.hstack([np.dot(c_B, self.A), np.zeros(self.A.shape[0])])
        zeroth_col = b
        zeroth_element = -1 * np.sum(b)

        rest = A.copy()

        ph1_tableau = np.block([
            [zeroth_element, zeroth_row],
            [zeroth_col.reshape(1, zeroth_col.shape[0]).T, rest]
        ])

        iters = 0

        while (ph1_tableau[0, 1:] < 0).any():

            j = np.where(ph1_tableau[0, 1:] < 0)[0][
                0]     # incoming basis direction
            theta = [i for i in range(1, ph1_tableau.shape[0]) if ph1_tableau[
                i, j + 1] > 0][0]
            for i in range(1, ph1_tableau.shape[0]):
                if ph1_tableau[i, j + 1] > 0 and ph1_tableau[i, 0] / ph1_tableau[i, j + 1] >= 0:
                    if ph1_tableau[i, 0] / ph1_tableau[i, j + 1] < ph1_tableau[theta, 0] / ph1_tableau[theta, j + 1]:
                        theta = i

            basic_columns[theta - 1] = j
            pivot_row = theta          # index of direction which will exit the basis matrix
            pivot_col = j + 1          # direction which will enter the basis

            ph1_tableau[pivot_row, :] = ph1_tableau[
                pivot_row, :] / ph1_tableau[pivot_row, pivot_col]

            for i in range(ph1_tableau.shape[0]):
                if i == pivot_row:
                    continue
                ph1_tableau[i, :] = ph1_tableau[i, :] - (ph1_tableau[i, pivot_col] / ph1_tableau[
                                                         pivot_row, pivot_col]) * ph1_tableau[pivot_row, :]

            iters += 1
            if iters == self.max_iter:
                raise RuntimeError(
                    "Cycling encountered! Method could not converge in max_iter = %d iterations. Terminating..." % (self.max_iter))

        if ph1_tableau[0, 0] > 0:
            raise RuntimeError("Given LP is infeasible!")

        elif ph1_tableau[0, 0] == 0:

            if (basic_columns < self.A.shape[1]).all():

                return ph1_tableau[1:, :self.A.shape[1] + 1], basic_columns

            else:

                while True:
                    av_inbasis_at = np.where(basic_columns >= self.A.shape[1])[
                        0].tolist()

                    if (ph1_tableau[av_inbasis_at[0] + 1, 1:self.A.shape[1] + 1] == 0).all():
                        ph1_tableau = np.delete(
                            ph1_tableau, (av_inbasis_at[0] + 1), axis=0)
                        self.A = np.delete(self.A, av_inbasis_at[0], axis=0)
                        self.b = np.delete(self.b, av_inbasis_at[0])
                        basic_columns = np.delete(
                            basic_columns, av_inbasis_at[0])

                    else:
                        pivot_row = av_inbasis_at[0] + 1
                        pivot_col = np.where(
                            ph1_tableau[pivot_row, 1:] != 0) + 1
                        ph1_tableau[pivot_row, :] = ph1_tableau[
                            pivot_row, :] / ph1_tableau[pivot_row, pivot_col]
                        for i in range(ph1_tableau.shape[0]):
                            if i == pivot_row:
                                continue
                            ph1_tableau[i, :] = ph1_tableau[i, :] - (ph1_tableau[i, pivot_col] / ph1_tableau[
                                                                     pivot_row, pivot_col]) * ph1_tableau[pivot_row, :]
                        basic_columns[av_inbasis_at[0]] = pivot_col - 1
                    av_inbasis_at = np.where(basic_columns >= self.A.shape[1])[
                        0].tolist()
                    if len(av_inbasis_at) == 0:
                        break

        return ph1_tableau[1:, :(self.A.shape[1] + 1)], basic_columns

    def run_phase2(self, tableau, basic_columns):
        self.tableau = tableau.copy()
        iters = 0
        while (tableau[0, 1:] < 0).any():
            j = np.where(tableau[0, 1:] < 0)[0][
                0]     # incoming basis direction
            theta = [i for i in range(1, tableau.shape[0]) if tableau[
                i, j + 1] > 0][0]
            for i in range(1, tableau.shape[0]):
                if tableau[i, j + 1] > 0 and tableau[i, 0] / tableau[i, j + 1] >= 0:
                    if tableau[i, 0] / tableau[i, j + 1] < tableau[theta, 0] / tableau[theta, j + 1]:
                        theta = i

            basic_columns[theta - 1] = j

            pivot_row = theta          # index of direction which will exit the basis matrix
            pivot_col = j + 1          # direction which will enter the basis

            tableau[pivot_row, :] = tableau[
                pivot_row, :] / tableau[pivot_row, pivot_col]

            for i in range(tableau.shape[0]):
                if i == pivot_row:
                    continue
                tableau[i, :] = tableau[
                    i, :] - (tableau[i, pivot_col] / tableau[pivot_row, pivot_col]) * tableau[pivot_row, :]

            iters += 1
            if iters == self.max_iter:
                raise RuntimeError(
                    "Method could not converge in max_iter = %d iterations. Terminating method...!\n\n" % (self.max_iter))

        self.solution = LPSolution(self.num_vars, {basic_columns[i]: tableau[1:, 0][
                                   i] for i in range(len(basic_columns))}, tableau[0, 0])

        return self.solution

    def run_simplex2(self):
        lower_half_tableau, initial_basis = self.run_phase1()
        b = self.b.copy()
        B = self.get_first_B(initial_basis)
        c_B = self.c[initial_basis]
        zeroth_element = -1 * np.dot(c_B, np.linalg.solve(B, b))
        zeroth_row = self.c - np.dot(c_B, np.dot(np.linalg.inv(B), self.A))

        tableau = np.vstack(
            (np.hstack((zeroth_element, zeroth_row)), lower_half_tableau))

        self.solution = self.run_phase2(tableau, initial_basis)

        return self.solution


def main():

    print("Solving for Flow 1\n")
    print("------------------------\n\n")

    constraints, obj = nf2PrimalLP("nf1.dat")
    splex = Simplex(14, 20, obj, constraints, 1000)
    sol = splex.run_simplex2()
    print("Max-Flow of the network is:", sol.obj)
    print("Detailed solution:\n")
    print(str(sol), "\n\n")

    constraints, obj = nf2DualLP("nf1.dat")
    splex = Simplex(10, 20, obj, constraints, 1000)
    sol = splex.run_simplex2()
    print("Min-cut of the network is:", sol.obj * (-1))
    print("Detailed solution:\n")
    print(str(sol), "\n\n")

    print("Solving for Flow 2\n")
    print("------------------------\n\n")

    constraints, obj = nf2PrimalLP("nf2.dat")
    splex = Simplex(39, 58, obj, constraints, 1000)
    sol = splex.run_simplex2()
    print("Max-Flow of the network is:", sol.obj)
    print("Detailed solution:\n")
    print(str(sol))

    constraints, obj = nf2DualLP("nf2.dat")
    splex = Simplex(29, 68, obj, constraints, 1000)
    sol = splex.run_simplex2()
    print("Min-cut of the network is:", sol.obj * (-1))
    print("Detailed solution:\n")
    print(str(sol))


if __name__ == '__main__':
    main()
