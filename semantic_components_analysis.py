import autograd.numpy as np
from autograd import grad
from autograd.numpy import sqrt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve

def _iteration(E, fixed_beta, method='minimizer_solver'):
    def objective_closure(e):
        def objective(beta):
            return -1 * 1/len(e) * np.sum(np.dot(e, beta))
        return objective

    def eq_orth_closure(beta_curr):
        def eq_orth(beta):
            return np.dot(beta_curr, beta)
        return eq_orth

    def eq_unit(beta):
        return sqrt(np.sum(np.power(beta, 2))) - 1

    def Lagrangian_closure(objective, eq_orth):
        def Lagrangian(X):
            'Augmented Lagrange function'
            beta, _lambda_orth, _lambda_unit = X
            return objective(beta) - np.dot(_lambda_orth, eq_orth(beta)) - _lambda_unit * eq_unit(beta)
        return Lagrangian

    def grad_Lagrangian_closure(e, beta_curr):
        def gradLagrangian(X):
            beta, _lambda_orth, _lambda_unit = X
            #print('beta_curr shape: ', beta_curr.shape)
            #print('beta shape: ', beta.shape)
            #print('_lambda_orth shape: ', _lambda_orth.shape)
            #print('_lambda_unit shape: ', _lambda_unit)
            dbeta = (1/len(e) * np.sum(e, axis=0)) - np.dot(beta_curr.T, _lambda_orth) - (_lambda_unit * 2 * beta)
            dlambda_orth = -1 * np.dot(beta_curr, beta)
            dlambda_unit = -2 * beta
            return dbeta, dlambda_orth, dlambda_unit
        return gradLagrangian

    def solve_Lagrangian_closure(grad_lagrangian, eq_orth, n_dims, n_fixed_betas):
        def solve_lagrangian(X):
            # scipy's fsolver function requires a one-dimensional
            # array, so we pack all of the arguments into X. Now
            # we must unpack them
            beta = X[:n_dims]
            _lambda_orth = X[n_dims:n_dims+n_fixed_betas]
            _lambda_unit = X[-1]
            dbeta, dlam_orth, dlam_unit = grad_lagrangian([
                beta, 
                _lambda_orth, 
                _lambda_unit
            ])
            # Reconcatenate into an array
            return np.concatenate([
                dbeta, 
                eq_orth(beta), 
                np.array([eq_unit(beta)])
            ])
        return solve_lagrangian

    def jacobian_closure(e):
        def jacobian(beta):
            return 1/len(e) * np.sum(e, axis=0) 
        return jacobian

    the_eq_orth = eq_orth_closure(fixed_beta)
    the_objective = objective_closure(E)
    the_jacobian = jacobian_closure(E)
    lagrangian = Lagrangian_closure(the_objective, the_eq_orth)
    grad_lagrangian = grad_Lagrangian_closure(E, fixed_beta)

    if method == 'root_solver':
        #grad_lagrangian = grad(lagrangian, 0)
        solve_lagrangian = solve_Lagrangian_closure(grad_lagrangian, the_eq_orth, len(fixed_beta[0]), len(fixed_beta))
        init = np.concatenate([np.array([1.]), np.ones(len(E[0]) + len(fixed_beta))])
        solution = fsolve(solve_lagrangian, init)
        #print(solution)
        return solution[:len(fixed_beta[0])], True
    elif method == 'minimizer_solver':
        solution = minimize(
            the_objective,
            np.concatenate([np.array([1.]), np.zeros(len(E[0])-1)]),
            jac=the_jacobian,
            options={
                'maxiter': 10000,
                'ftol': 0.0001
            },
            constraints=[
                {'type': 'eq', 'fun': the_eq_orth},
                {'type': 'eq', 'fun': eq_unit}
                #{'type': 'ineq', 'fun': bound_one}
            ]
        )
        print(solution)
        print(np.linalg.norm(solution.x))
        print()
        return solution.x, solution.success

def SCA(E, method='minimizer_solver'):
    print('Normalizing input vectors...')
    E = np.array([
        e/np.linalg.norm(e)
        for e in E
    ])
    print('done.')
    beta_init = np.sum(E, axis=0) / np.linalg.norm(np.sum(E, axis=0))
    curr_fixed_beta = [beta_init]
    statuses = [True]
    for i in range(1,len(beta_init)):
        print(f'Solving dimension {i}...')
        new_beta, status = _iteration(E, np.array(curr_fixed_beta), method=method)
        statuses.append(status)
        curr_fixed_beta.append(new_beta)
    return np.array(curr_fixed_beta), statuses

def main():
    E = np.array([
        [1.,2.,3.,4.],
        [2.,4.,3.,4.],
        [1.,2.,2.,1.]
    ])
    BETA_FIXED = np.array([
        #[0.,0.,0.,1.],
        [0.,0.,1.,0.]
    ])
    #print(_iteration(E, BETA_FIXED))

    sca, statuses = SCA(E, method='root_solver')
    print(sca)
    for i1, x1 in enumerate(sca):
        for i2, x2 in enumerate(sca):
            print('element {},{}: {}'.format(i1, i2, np.dot(x1, x2)))

if __name__ == '__main__':
    main()
