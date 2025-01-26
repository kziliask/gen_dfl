import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel

import torch

class knapsackModelRel(pyepo.model.grb.knapsackModel):
    """
    This class is relaxed optimization model for knapsack problem.
    """

    def _getModel(self):
        """
        A method to build Gurobi
        """
        # ceate a model
        m = gp.Model("knapsack")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addMVar(self.items, name="x", ub=1, vtype=GRB.CONTINUOUS)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x

    def relax(self):
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")
    
class ExpectedKnapsackModel(pyepo.model.grb.knapsackModel):
    """
    This class is optimization model for knapsack problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray / list): Weights of items
        capacity (np.ndarray / listy): Total capacity
        items (list): List of item index
    """

    def __init__(self, weights, capacity):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
        """
        self.weights = np.array(weights)
        self.capacity = np.array(capacity)
        self.items = self.weights.shape[1]
        super().__init__(weights, capacity)

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("knapsack")
        # varibles
        x = m.addMVar(self.items, name="x", ub=1, vtype=GRB.CONTINUOUS)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x
    
    def setObj(self, c_samples, alpha=1):
        '''
        min_w E_c[f(w, c)]
        c_samples: [num_samples, num_items]
        '''
        obj = gp.quicksum(
            (1/c_samples.shape[0]) * np.array(c_samples[i]) @ self.x
            for i in range(c_samples.shape[0])
        )
        self._model.setObjective(obj)
    
    def regret_loss_batch(self, xs, c_samples, c_trues, alpha=1):
        '''
        Compute the regret loss for a batch of solutions
        c_samples: [batch_size=1, num_samples, num_items]
        c_trues: [batch_size=1, num_samples, num_items]
        '''
        batch_size = xs.shape[0]
        loss = 0
        
        for c_sample, c_true in zip(c_samples, c_trues):
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            # Convert sol to tensor if it's not already
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs = torch.matmul(c_true, sol_tensor)
            # Sort the objectives
            objs_sorted, _ = torch.sort(objs)
            # Take the worst alpha fraction
            m = int(alpha * len(objs))
            regret1 = torch.mean(objs_sorted[:m])
            
            self.setObj(c_true.detach())
            sol_true, _ = self.solve()
            # Convert sol_true to tensor if it's not already
            sol_true_tensor = torch.tensor(sol_true, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs_true = torch.matmul(c_true, sol_true_tensor)
            # Sort the objectives
            objs_true_sorted, _ = torch.sort(objs_true)
            # Take the worst alpha fraction
            m = int(alpha * len(objs_true))
            regret2 = torch.mean(objs_true_sorted[:m])
            
            loss += torch.abs(regret1 - regret2)
            
        return loss / batch_size
    
    def obj_eval(self, xs, c_samples, c_trues, alpha=1):
        '''
        Compute the objective value for a batch of solutions
        c_samples: [batch_size=1, num_samples, num_items]
        c_trues: [batch_size=1, num_samples, num_items]
        '''
        batch_size = xs.shape[0]
        loss = 0
        
        for c_sample, c_true in zip(c_samples, c_trues):
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            # Convert sol to tensor if it's not already
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs = torch.matmul(c_true, sol_tensor)
            # Sort the objectives
            objs_sorted, _ = torch.sort(objs)
            # Take the worst alpha fraction
            m = int(alpha * len(objs))
            loss += torch.mean(objs_sorted[:m])
            
        return loss / c_samples.shape[0]
        
class ExpectedShortestPathModel(pyepo.model.grb.shortestPathModel):
    """
    This class is optimization model for shortest path problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        grid (tuple of int): Size of grid network
        arcs (list): List of arcs
    """
    def __init__(self, grid):
        """
        Args:
            grid (tuple of int): size of grid network
        """
        self.grid = grid
        self.arcs = self._getArcs()
        super().__init__(grid)
        
        
    def setObj(self, c_samples, alpha=1):
        '''
        min_w E_c[f(w, c)]
        c_samples: [batch_size, num_samples, num_items]
        '''
        if isinstance(self.x, gp.MVar):
            obj = gp.quicksum(
                (1/c_samples.shape[0]) * np.array(c_samples[i]) @ self.x
                for i in range(c_samples.shape[0])
            )
        else:
            obj = gp.quicksum(
                (1/c_samples.shape[0]) * gp.quicksum(c_samples[i][j] * self.x[k] for j, k in enumerate(self.x))
                for i in range(c_samples.shape[0])
            )
        self._model.setObjective(obj)
        
    def regret_loss_batch(self, xs, c_samples, c_trues, alpha=1):
        '''
        Compute the regret loss for a batch of solutions
        c_samples: [batch_size=1, num_samples, num_items]
        c_trues: [batch_size=1, num_samples, num_items]
        '''
        batch_size = xs.shape[0]
        loss = 0
        
        for c_sample, c_true in zip(c_samples, c_trues):
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            # Convert sol to tensor if it's not already
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs = torch.matmul(c_true, sol_tensor)
            # Sort the objectives
            objs_sorted, _ = torch.sort(objs)
            # Take the worst alpha fraction
            m = int(alpha * len(objs))
            regret1 = torch.mean(objs_sorted[:m])
            
            self.setObj(c_true.detach())
            sol_true, _ = self.solve()
            # Convert sol_true to tensor if it's not already
            sol_true_tensor = torch.tensor(sol_true, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs_true = torch.matmul(c_true, sol_true_tensor)
            # Sort the objectives
            objs_true_sorted, _ = torch.sort(objs_true)
            # Take the worst alpha fraction
            m = int(alpha * len(objs_true))
            regret2 = torch.mean(objs_true_sorted[:m])
            
            loss += torch.abs(regret1 - regret2)
            
        return loss / batch_size
    
    def obj_eval(self, xs, c_samples, c_trues, alpha=1):
        '''
        Compute the objective value for a batch of solutions
        c_samples: [batch_size=1, num_samples, num_items]
        c_trues: [batch_size=1, num_samples, num_items]
        '''
        batch_size = xs.shape[0]
        loss = 0
        
        for c_sample, c_true in zip(c_samples, c_trues):
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            # Convert sol to tensor if it's not already
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs = torch.matmul(c_true, sol_tensor)
            # Sort the objectives
            objs_sorted, _ = torch.sort(objs)
            # Take the worst alpha fraction
            m = int(alpha * len(objs))
            loss += torch.mean(objs_sorted[:m])
            
        return loss / c_samples.shape[0]

    
class ExpectedPortfolioModel(pyepo.model.grb.portfolioModel):
    """
    This class is an optimization model for portfolio problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_assets (int): number of assets
        covariance (numpy.ndarray): covariance matrix of the returns
        risk_level (float): risk level
    """
    def __init__(self, num_assets, covariance, gamma=2.25, risk=False):
        """
        Args:
            num_assets (int): number of assets
            covariance (numpy.ndarray): covariance matrix of the returns
            gamma (float): if to include the quadratic loss for MIP approx.
        """
        self.num_assets = num_assets
        self.covariance = covariance
        self.risk_level = self._getRiskLevel(gamma)
        self.risk = risk
        super().__init__(num_assets, covariance, gamma)
        
    def setObj(self, c_samples, alpha=1):
        '''
        min_w E_c[f(w, c)]
        '''
        if self.risk is True:
            obj = gp.quicksum(
                (1/c_samples.shape[0]) * (np.array(c_samples[i]) @ self.x)
                for i in range(c_samples.shape[0]) 
            ) + self.x @ self.covariance @ self.x
        else:
            obj = gp.quicksum(
                (1/c_samples.shape[0]) * np.array(c_samples[i]) @ self.x
                for i in range(c_samples.shape[0])
            )
        self._model.setObjective(obj)
        
    def regret_loss(self, xs, contextual, contextual_gt, alpha=1, num_samples=200):
        """
        Expected value of the worst alpha% tail of the realizations in c_samples
        E_Cvar[f(w^\star, c)]
        """
        # [batch_size, num_samples, num_items]
        
        c_samples = contextual.sample(num_samples, xs)
        c_trues = contextual_gt.sample(num_samples, xs).detach()
        
        batch_size = xs.shape[0]
        loss = 0 

        for c_sample, c_true in zip(c_samples, c_trues):
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            # Convert sol to tensor if it's not already
            # sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs = torch.matmul(c_true, sol)
            # Sort the objectives
            objs_sorted, _ = torch.sort(objs)
            # Take the worst alpha fraction
            m = int(alpha * len(objs))
            loss += torch.mean(objs_sorted[-m:])
            
        return loss / batch_size
    
    def regret_loss_batch(self, xs, c_samples, c_trues, alpha=1):
        '''
        Compute the regret loss for a batch of solutions
        c_samples: [batch_size=1, num_samples, num_items]
        c_trues: [batch_size=1, num_samples, num_items]
        '''
        batch_size = xs.shape[0]
        loss = 0
        
        for c_sample, c_true in zip(c_samples, c_trues):
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            # Convert sol to tensor if it's not already
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs = torch.matmul(c_true, sol_tensor)
            # Sort the objectives
            objs_sorted, _ = torch.sort(objs)
            # Take the worst alpha fraction
            m = int(alpha * len(objs))
            regret1 = torch.mean(objs_sorted[:m])
            
            self.setObj(c_true.detach())
            sol_true, _ = self.solve()
            # Convert sol_true to tensor if it's not already
            sol_true_tensor = torch.tensor(sol_true, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs_true = torch.matmul(c_true, sol_true_tensor)
            # Sort the objectives
            objs_true_sorted, _ = torch.sort(objs_true)
            # Take the worst alpha fraction
            m = int(alpha * len(objs_true))
            regret2 = torch.mean(objs_true_sorted[:m])
            
            loss += torch.abs(regret1 - regret2)
            
        return loss / batch_size
    
    def obj_eval(self, xs, c_samples, c_trues, alpha=1):
        '''
        Compute the objective value for a batch of solutions
        c_samples: [batch_size=1, num_samples, num_items]
        c_trues: [batch_size=1, num_samples, num_items]
        '''
        batch_size = xs.shape[0]
        loss = 0
        
        for c_sample, c_true in zip(c_samples, c_trues):
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            # Convert sol to tensor if it's not already
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs = torch.matmul(c_true, sol_tensor)
            # Sort the objectives
            objs_sorted, _ = torch.sort(objs)
            # Take the worst alpha fraction
            m = int(alpha * len(objs))
            loss += torch.mean(objs_sorted[:m])
            
        return loss / c_samples.shape[0]
    
class ExpectedSyntheticNonlinearModel(optGrbModel):
    """
    This class is an optimization model for synthetic nonlinear problem
    Use quadratic loss for approx.
    """
    def __init__(self, num_items):
        self.num_items = num_items
        super().__init__()

    def _getModel(self):
        """
        A method to build Gurobi model
        """
        m = gp.Model("nonlinear")
        # variables
        x = m.addMVar(self.num_items, name="x", vtype=GRB.CONTINUOUS)
        # sense
        m.modelSense = GRB.MINIMIZE
        return m, x
    
    def setObj(self, c_samples, alpha=1):
        '''
        min_w E_c[f(w, c)]
        '''
        # T = 
        # obj = gp.quicksum(
        #     (1/c_samples.shape[0]) *\
        #           gp.quicksum(np.array(c_samples[i]) @ self.x + 0.5 * (self.x[t] - expected_values[t]) * (self.x[t] - expected_values[t]) for t in range()) 
        #     for i in range(c_samples.shape[0])
        # )
        # self._model.setObjective(obj)
        pass 

    def regret_loss_batch(self, xs, c_samples, c_trues, alpha=1):
        '''
        Compute the regret loss for a batch of solutions
        c_samples: [batch_size=1, num_samples, num_items]
        c_trues: [batch_size=1, num_samples, num_items]
        '''
        batch_size = xs.shape[0]
        loss = 0
        
        for c_sample, c_true in zip(c_samples, c_trues):
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            # Convert sol to tensor if it's not already
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs = torch.matmul(c_true, sol_tensor)
            # Sort the objectives
            objs_sorted, _ = torch.sort(objs)
            # Take the worst alpha fraction
            m = int(alpha * len(objs))
            regret1 = torch.mean(objs_sorted[:m])
            
            self.setObj(c_true.detach())
            sol_true, _ = self.solve()
            # Convert sol_true to tensor if it's not already
            sol_true_tensor = torch.tensor(sol_true, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs_true = torch.matmul(c_true, sol_true_tensor)
            # Sort the objectives
            objs_true_sorted, _ = torch.sort(objs_true)
            # Take the worst alpha fraction
            m = int(alpha * len(objs_true))
            regret2 = torch.mean(objs_true_sorted[:m])
            
            loss += torch.abs(regret1 - regret2)
            
        return loss / batch_size
    
    def obj_eval(self, xs, c_samples, c_trues, alpha=1):
        '''
        Compute the objective value for a batch of solutions
        c_samples: [batch_size=1, num_samples, num_items]
        c_trues: [batch_size=1, num_samples, num_items]
        '''
        batch_size = xs.shape[0]
        loss = 0
        
        for c_sample, c_true in zip(c_samples, c_trues):
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            # Convert sol to tensor if it's not already
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs = torch.matmul(c_true, sol_tensor)
            # Sort the objectives
            objs_sorted, _ = torch.sort(objs)
            # Take the worst alpha fraction
            m = int(alpha * len(objs))
            loss += torch.mean(objs_sorted[:m])
            
        return loss / c_samples.shape[0]
    
class ExpectedDemandResponse(optGrbModel):
    """
    This class is an optimization model for energy scheduling.

    Attributes:
        _model (model): a Pyomo model
        plo_t (np.ndarray / list): lower bound of the demand
        psch_t (np.ndarray / list): scheduled  demand
        pup_t (np.ndarray / list): upper bound of the demand
    """

    def __init__(self, plo_t, psch_t, pup_t):
        """
        Args:
            plo_t (np.ndarray / list): lower bound of the demand
            psch_t (np.ndarray / list): scheduled  demand
            pup_t (np.ndarray / list): upper bound of the demand
        """
        self.plo_t = np.array(plo_t)
        self.psch_t = np.array(psch_t)
        self.pup_t = np.array(pup_t)
        super().__init__()

    def _getModel(self):
        # create a model
        m = gp.Model("Energy")
        # variables
        x = m.addMVar(24, lb=self.plo_t, ub=self.pup_t, name="x")
        # constr
        m.addConstr(gp.quicksum(x) == gp.quicksum(self.psch_t))
        return m, x
    
    def setObj(self, c_samples):
        '''
        min_x E_c[f(x, c)]
        '''
        obj = gp.quicksum(
            (1/c_samples.shape[0]) * np.array(c_samples[i]) @ self.x
            for i in range(c_samples.shape[0])
        )
        self._model.setObjective(obj)
        
    def obj_eval(self, xs, c_samples, c_trues, alpha=1):
        '''
        Compute the objective value for a batch of solutions
        c_samples: [batch_size, num_samples, num_items]
        c_trues: [batch_size, num_samples, num_items]
        '''
        
        loss = 0
        for c_sample, c_true in zip(c_samples, c_trues):
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            # Convert sol to tensor if it's not already
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)
            # Compute all objectives at once: [n, d] @ [d] -> [n]
            objs = torch.matmul(c_true, sol_tensor)
            # Sort the objectives
            objs_sorted, _ = torch.sort(objs)
            # Take the worst alpha fraction
            m = int(alpha * len(objs))
            loss += torch.mean(objs_sorted[-m:])
            
        return loss / c_samples.shape[0]
        
class AverageRegretModel(pyepo.model.grb.knapsackModel):
    """
    This class is optimization model for knapsack problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray / list): Weights of items
        capacity (np.ndarray / listy): Total capacity
        items (list): List of item index
    """

    def __init__(self, weights, capacity):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
        """
        self.weights = np.array(weights)
        self.capacity = np.array(capacity)
        self.items = self.weights.shape[1]
        super().__init__(weights, capacity)

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("knapsack")
        # varibles
        x = m.addMVar(self.items, name="x", ub=1, vtype=GRB.CONTINUOUS)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(self.weights @ x <= self.capacity)
        return m, x
    
    def setObj(self, c_samples):
        obj = gp.quicksum(
            (1/c_samples.shape[0]) * c_samples[i] @ self.x
            for i in range(c_samples.shape[0])
        )
        self._model.setObjective(obj)

class AverageUnderGroundingModel(pyepo.model.grb.knapsackModel):
    """
    This class is a 'knapsack-like' optimization model for an undergrounding/hardening problem.
    It handles scenario-based average objective.

    Attributes:
        _model (gurobipy.Model): Gurobi model
        customer (np.ndarray or list): e.g. customer weighting per city
        capacity (float or int): total capacity (max number of cities to be hardened)
        items (int): number of items/cities
    """

    def __init__(self, customer, capacity):
        """
        Args:
            customer (np.ndarray or list): city-level 'weights' or # of customers
            capacity (float or int): total capacity (# of cities that can be chosen)
        """
        self.customer = np.array(customer, dtype=float)
        self.capacity = capacity
        self.items = len(self.customer)
        # Inheriting from knapsackModel: pass the same shape 
        # (the code expects 'weights' and 'capacity'). 
        # We'll call it as if 'weights' is shape (1, items).
        super().__init__(self.customer[np.newaxis, :], capacity)

    def _getModel(self):
        """
        Build a Gurobi model: x[i] in [0,1], sum(x) <= capacity.
        We set sense=MAXIMIZE. 
        (Switch vtype=GRB.BINARY if you want a strict city selection.)
        """
        m = gp.Model("undergrounding")
        m.Params.outputFlag = 0  # turn off Gurobi logging if desired

        # x: how many fraction of city i we 'underground'? 
        # If you want a pure binary selection, use vtype=GRB.BINARY:
        x = m.addMVar(self.items, name="x", ub=1, vtype=GRB.CONTINUOUS)

        m.modelSense = GRB.MAXIMIZE

        # Constraint: sum(x) <= capacity
        m.addConstr(x.sum() <= self.capacity)

        return m, x

    def setObj(self, c_samples):
        """
        Sets Gurobi objective to the average over scenarios in c_samples. 
        Suppose c_samples has shape (num_scenarios, num_cities).

        We'll do: 
          objective = (1/num_scenarios) * sum_{s} [ (c_samples[s,:] * customer) dot x ].

        The user can interpret c_samples[s,i] as some cost or benefit 
        for city i in scenario s, multiplied by city i's 'customer' factor.
        """
        num_scen = c_samples.shape[0]
        # Build Gurobi expression
        obj_expr = 0.0

        for s in range(num_scen):
            # Convert to numpy if needed
            cost_vec = np.array(c_samples[s]) * self.customer
            # cost_vec @ x => sum_i cost_vec[i] * x[i]
            obj_expr += (1.0 / num_scen) * (cost_vec @ self.x)

        self._model.setObjective(obj_expr, GRB.MAXIMIZE)


class ExpectedUnderGroundingModel(pyepo.model.grb.knapsackModel):
    """
    A 'knapsack-style' model for undergrounding (or city hardening).

    Attributes:
        _model (GurobiPy model): Gurobi model
        customer (np.ndarray): For each city (item), e.g. # of customers or weighting
        capacity (float or int): Max number of items/cities that can be chosen (like knapsack capacity)
        items (int): number of items
    """

    def __init__(self, customer, capacity):
        """
        Args:
            customer (np.ndarray or list): 'weight' or # of customers for each city.
            capacity (float or int): total capacity (max chosen items).
        """
        self.customer = np.array(customer, dtype=float)
        self.capacity = capacity
        self.items = len(self.customer)   # or self.customer.shape[0]
        super().__init__(self.customer, capacity)

    def _getModel(self):
        """
        Build Gurobi model: continuous or binary knapsack with capacity on x.sum().
        """
        m = gp.Model("knapsack")
        # Turn off Gurobi output if desired:
        m.Params.outputFlag = 0

        # x: let's assume continuous or binary. If truly knapsack, maybe vtype=GRB.BINARY
        x = m.addMVar(self.items, name="x", ub=1, vtype=GRB.CONTINUOUS)

        # We want to maximize
        m.modelSense = GRB.MAXIMIZE

        # Constraint: sum(x) <= capacity
        m.addConstr(x.sum() <= self.capacity)

        return m, x

    def setObj(self, c_samples, alpha=1):
        """
        Sets the Gurobi objective to the average of
           ( cost[i,:] * self.customer ) dot x
        over all scenarios i in c_samples.

        c_samples: shape (num_samples, num_items)
        """
        num_scen = c_samples.shape[0]
        # We'll build up a Gurobi expression
        obj_expr = 0.0

        for i in range(num_scen):
            # Multiply each item cost by the city's 'customer' factor
            # c_samples[i] is shape (num_items,)
            # (c_samples[i] * self.customer) is also shape (num_items,)
            cost_vec = np.array(c_samples[i]) * self.customer  # shape (items,)

            # Then cost_vec @ self.x is that scenario's total cost
            # We'll weight it by (1/num_scen)
            obj_expr += (1.0 / num_scen) * (cost_vec @ self.x)

        self._model.setObjective(obj_expr, GRB.MAXIMIZE)


    def regret_loss_batch(self, xs, c_samples, c_trues, alpha=1):
        """
        c_samples, c_trues shape: [batch_size=1, num_scenarios, num_items].
        We'll incorporate `self.customer` in the objective computations below.
        """
        batch_size = xs.shape[0]
        loss = 0

        # We'll build a torch version of self.customer for multiplication
        cust_t = torch.from_numpy(self.customer).to(c_samples.device).float()

        for c_sample, c_true in zip(c_samples, c_trues):
            # c_sample, c_true each shape: (num_scenarios, num_items)

            # 1) Solve with c_sample
            self.setObj(c_sample.detach())  # Gurobi objective
            sol, _ = self.solve()           # Solve
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)

            # Evaluate that solution under the 'true' costs:
            # Each scenario row => c_true[s,:], and then multiply by customer
            # So we do (c_true[s,:] * cust_t) dot sol_tensor
            # We'll gather them all at once
            # => shape (num_scenarios,)
            objs = torch.matmul(c_true * cust_t, sol_tensor)
            objs_sorted, _ = torch.sort(objs)
            m = int(alpha * len(objs))
            regret1 = torch.mean(objs_sorted[:m])

            # 2) Solve with c_true (the real cost)
            self.setObj(c_true.detach())
            sol_true, _ = self.solve()
            sol_true_tensor = torch.tensor(sol_true, dtype=c_sample.dtype, device=c_sample.device)

            # Evaluate that 'true' solution under c_true again
            objs_true = torch.matmul(c_true * cust_t, sol_true_tensor)
            objs_true_sorted, _ = torch.sort(objs_true)
            m = int(alpha * len(objs_true))
            regret2 = torch.mean(objs_true_sorted[:m])

            # Accumulate absolute difference
            loss += torch.abs(regret1 - regret2)

        return loss / batch_size

    def obj_eval(self, xs, c_samples, c_trues, alpha=1):
        """
        Evaluate the objective (e.g. CVaR alpha) for solutions obtained from c_samples,
        measured under c_trues. Also incorporate self.customer in the objectives.
        """
        batch_size = xs.shape[0]
        loss = 0

        cust_t = torch.from_numpy(self.customer).to(c_samples.device).float()

        for c_sample, c_true in zip(c_samples, c_trues):
            # Solve the model with c_sample
            self.setObj(c_sample.detach())
            sol, _ = self.solve()
            sol_tensor = torch.tensor(sol, dtype=c_sample.dtype, device=c_sample.device)

            # Evaluate under c_true
            objs = torch.matmul(c_true * cust_t, sol_tensor)
            objs_sorted, _ = torch.sort(objs)
            m = int(alpha * len(objs))
            loss += torch.mean(objs_sorted[:m])

        return loss / c_samples.shape[0]

        
if __name__ == "__main__":

    model = pyepo.model.grb.ShortesPathModel()