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
        c_samples: [batch_size, num_samples, num_items]
        c_trues: [batch_size, num_samples, num_items]
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
        c_samples: [batch_size, num_samples, num_items]
        c_trues: [batch_size, num_samples, num_items]
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
    def __init__(self, num_assets, covariance, gamma=2.25):
        """
        Args:
            num_assets (int): number of assets
            covariance (numpy.ndarray): covariance matrix of the returns
            gamma (float): risk level parameter
        """
        self.num_assets = num_assets
        self.covariance = covariance
        self.risk_level = self._getRiskLevel(gamma)
        super().__init__(num_assets, covariance, gamma)
        
    def setObj(self, c_samples, alpha=1):
        '''
        min_w E_c[f(w, c)]
        '''
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

if __name__ == "__main__":

    model = pyepo.model.grb.ShortesPathModel()