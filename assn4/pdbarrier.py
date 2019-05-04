# Name: Aditya Chetan
# Roll No.: 2016217


import numpy as np
import numpy.linalg as npla


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

    primal_constraints = np.vstack((primal_constraints[:numedges], primal_constraints[
                                   numedges + 1:numedges + numvertices + 1]))

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
        sol += "\tSolution to the LP is as follows:\n\n"
        sol += "optim\t:=\t" + str(self.obj) + "\n\n"
        for i in range(self.num_vars):
            if i in self.var_vals:
                sol += "x_" + str(i + 1) + "*\t:=\t" + \
                    str(self.var_vals[i]) + "\n"
            else:
                sol += "x_" + str(i + 1) + "*\t:=\t" + str(0.0) + "\n"
        return sol


class PrimalDual(object):

    def __init__(self, num_eq_constraints, num_vars, c, constraints=None, tol=10**-20, A=None, b=None):
        self.num_eq_constraints = num_eq_constraints
        self.num_vars = num_vars
        self.A = A
        self.b = b
        self.tol = tol
        self.constraints = constraints
        if constraints is not None:
            self.A = constraints[:self.num_eq_constraints, :-1]
            self.b = constraints[:self.num_eq_constraints, -1]
        self.c = c
        if A is not None:
            self.num_vars = A.shape[1]
        self.x = None
        self.s = None
        self.lamda = None
        self.obj = 0

    def fetch_constraints(self, filename):
        constraints = np.loadtxt(filename)
        self.A = constraints[:self.num_eq_constraints, :-1]
        self.b = constraints[:self.num_eq_constraints, -1]

    def run_PDBarrier(self):

        if self.A is None or self.b is None:
            raise RuntimeError("Please fetch the constraints first!")

        A = self.A.copy()
        b = self.b.copy()
        c = self.c.copy()

        np.random.seed(2010)
        x = np.random.random(A.shape[1])
        lamda = np.random.random(A.shape[0])
        s = np.random.random(A.shape[1])

        print(x.shape, lamda.shape)

        n = A.shape[1]
        mu = np.dot(x, s) / n
        sigma = 1 - (1 / np.sqrt(n))
        e = np.ones(x.shape[0])
        eta = 0.9999

        while mu > self.tol:
            X = np.diag(x)
            S = np.diag(s)
            I = np.eye(x.shape[0])
            bmatrix = np.block([
                [np.zeros((A.shape[1], A.shape[1])), A.T, I],
                [A, np.zeros((A.shape[0], A.shape[0])),
                 np.zeros((A.shape[0], x.shape[0]))],
                [S, np.zeros((S.shape[0], A.shape[0])), X]
            ])
            rhs = -1 * np.hstack((
                np.dot(A.T, lamda) + s - c,
                np.dot(A, x) - b,
                np.dot(np.dot(X, S), e) - (sigma * mu * e)
            ))
            delta = np.linalg.solve(bmatrix, rhs)
            delx = delta[:x.shape[0]]
            dell = delta[x.shape[0]:-s.shape[0]]
            dels = delta[-s.shape[0]:]
            alphax = 1.0

            if delx[delx < 0].shape[0] > 0:
                alphax = np.minimum(
                    1.0, eta * np.min(-1 * x[delx < 0] / delx[delx < 0]))

            alphas = 1.0
            if dels[dels < 0].shape[0] > 0:
                alphas = np.minimum(
                    1.0, eta * np.min(-1 * s[dels < 0] / dels[dels < 0]))
            x = x + (alphax * delx)
            s = s + (alphas * dels)
            lamda = lamda + (alphas * dell)
            mu = np.dot(s, x) / n

        self.x = x.copy()
        self.s = s.copy()
        self.lamda = lamda.copy()

#         self.x[self.x <= self.tol] = 0
#         self.s[self.s <= self.tol] = 0
#         self.lamda[self.lamda <= self.tol] = 0

        self.obj = np.dot(c, x)

        var_vals = {i: self.x[i] for i in range(self.num_vars)}
        return LPSolution(self.num_vars, var_vals, self.obj)


def main():

    print("Solving for Flow 1\n")
    print("------------------------\n\n")
    constraints, c = nf2PrimalLP("nf1.dat")
    pd = PrimalDual(14, 10, c, constraints, 10**-35)
    sol = pd.run_PDBarrier()
    print("Max-Flow of the network is:", sol.obj * (-1))
    print("Detailed solution:\n")
    print(str(sol), "\n\n")

    constraints, c = nf2DualLP("nf1.dat")
    pd = PrimalDual(10, 24, c, constraints, 10**-35)
    sol = pd.run_PDBarrier()
    print("Min-cut of the network is:", sol.obj)
    print("Detailed solution:\n")
    print(str(sol), "\n\n")

    print("Solving for Flow 2\n")
    print("------------------------\n\n")
    constraints, c = nf2PrimalLP("nf2.dat")
    pd = PrimalDual(39, 58, c, constraints, 10**-30)
    sol = pd.run_PDBarrier()
    print("Max-Flow of the network is:", sol.obj * (-1))
    print("Detailed solution:\n")
    print(str(sol))

    constraints, c = nf2DualLP("nf2.dat")
    pd = PrimalDual(29, 68, c, constraints, 10**-30)
    sol = pd.run_PDBarrier()
    print("Min-cut of the network is:", sol.obj)
    print("Detailed solution:\n")
    print(str(sol))


if __name__ == '__main__':
    main()
