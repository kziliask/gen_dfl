#!/usr/bin/env python
# coding: utf-8
"""
Noise contrastive estimation loss function
"""

import numpy as np
import torch
from torch import nn

from pyepo import EPO
# from abcmodule import optModule
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optDataset import optDataset
from pyepo.func.utlis import _solveWithObj4Par, _solve_in_pass
from pyepo.model.opt import optModel

from abc import abstractmethod
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool

class optModule(nn.Module):
    """
        An abstract module for the learning to rank losses, which measure the difference in how the predicted cost
        vector and the true cost vector rank a pool of feasible solutions.
    """
    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        # number of processes
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        self.processes = mp.cpu_count() if not processes else processes
        # single-core
        if self.processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(self.processes)
        print("Num of cores: {}".format(self.processes))
        # solution pool
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        self.solpool = None
        if self.solve_ratio < 1: # init solution pool
            if not isinstance(dataset, optDataset): # type checking
                raise TypeError("dataset is not an optDataset")
            self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate
        # reduction
        self.reduction = reduction

    @abstractmethod
    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        # convert tensor
        pass

    def _update_solution_pool(self, sol):
        """
        Add new solutions to solution pool
        """
        # add into solpool
        self.solpool = np.concatenate((self.solpool, sol))
        # remove duplicate
        self.solpool = np.unique(self.solpool, axis=0)


class NCE(optModule):
    """
    An autograd module for noise contrastive estimation as surrogate loss
    functions, based on viewing suboptimal solutions as negative examples.

    For the NCE, the cost vector needs to be predicted from contextual data and
    maximizes the separation of the probability of the optimal solution.

    Thus allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://www.ijcai.org/proceedings/2021/390>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        pred_cost: [batch_size, num_items]
        true_sol: [batch_size, num_items]
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            # add into solpool
            self._update_solution_pool(sol)
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get current obj
        # print(solpool.shape, pred_cost.shape, true_sol.shape)
        # obj_cp = torch.einsum("bd,bd->b", pred_cost.squeeze(), true_sol).unsqueeze(1)
        obj_cp = torch.matmul(pred_cost.squeeze(), true_sol).unsqueeze(0).mean(1)
        
        # get obj for solpool
        # objpool_cp = torch.einsum("bd,nd->bn", pred_cost, solpool)
        objpool_cp = torch.einsum('md,nd->mn', pred_cost.squeeze(), solpool).mean(axis=0).unsqueeze(0)
       
        # get loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = (obj_cp - objpool_cp).mean(axis=1)
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = (objpool_cp - obj_cp).mean(axis=1)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss


class contrastiveMAP(optModule):
    """
    An autograd module for Maximum A Posterior contrastive estimation as
    surrogate loss functions, which is an efficient self-contrastive algorithm.

    For the MAP, the cost vector needs to be predicted from contextual data and
    maximizes the separation of the probability of the optimal solution.

    Thus, it allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://www.ijcai.org/proceedings/2021/390>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate

    def forward(self, pred_cost, true_sol, alpha=1):
        """
        Forward pass
        pred_cost: [batch_size (1), num_samples, num_items]
        true_sol: [num_items]
        """
        # print(pred_cost.shape, true_sol.shape)
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            # add into solpool
            self._update_solution_pool(sol)
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get current obj
        # obj_cp = torch.einsum("bd,bd->b", pred_cost, true_sol).unsqueeze(1)
        obj_cp = torch.matmul(pred_cost.squeeze(), true_sol).unsqueeze(0) # [1, num_samples]
        
        m = int(alpha * obj_cp.shape[1])
        obj_cp, _ = torch.sort(obj_cp, dim=1)
        if self.optmodel.modelSense == EPO.MINIMIZE:
            obj_cp = obj_cp[:, -m:]
        else:
            obj_cp = obj_cp[:, :m]
        obj_cp = obj_cp.mean(axis=1)
        
        # get obj for solpool
        # objpool_cp = torch.einsum("bd,nd->bn", pred_cost, solpool)
        # [num_samples, num_sols]
        objpool_cp = torch.einsum('md,nd->mn', pred_cost.squeeze(), solpool)#.mean(axis=0).unsqueeze(0)
        objpool_cp, _ = torch.sort(objpool_cp, dim=0)
        
        if self.optmodel.modelSense == EPO.MINIMIZE:
            objpool_cp = objpool_cp[-m:, :]
        else:
            objpool_cp = objpool_cp[:m, :]
        objpool_cp = objpool_cp.mean(0).unsqueeze(0)
        
        # get loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss, _ = (obj_cp - objpool_cp).max(axis=1)
        if self.optmodel.modelSense == EPO.MAXIMIZE:
            loss, _ = (objpool_cp - obj_cp).max(axis=1)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss
    
class NCEPred(optModule):
    """
    An autograd module for noise contrastive estimation as surrogate loss
    functions, based on viewing suboptimal solutions as negative examples.

    For the NCE, the cost vector needs to be predicted from contextual data and
    maximizes the separation of the probability of the optimal solution.

    Thus allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://www.ijcai.org/proceedings/2021/390>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            # add into solpool
            self._update_solution_pool(sol)
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get current obj
        obj_cp = torch.einsum("bd,bd->b", pred_cost, true_sol).unsqueeze(1)
        # get obj for solpool
        objpool_cp = torch.einsum("bd,nd->bn", pred_cost, solpool)
        # get loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss = (obj_cp - objpool_cp).mean(axis=1)
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            loss = (objpool_cp - obj_cp).mean(axis=1)
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss


class contrastiveMAPPred(optModule):
    """
    An autograd module for Maximum A Posterior contrastive estimation as
    surrogate loss functions, which is an efficient self-contrastive algorithm.

    For the MAP, the cost vector needs to be predicted from contextual data and
    maximizes the separation of the probability of the optimal solution.

    Thus, it allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://www.ijcai.org/proceedings/2021/390>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data, usually this is simply the training set
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # solution pool
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.solpool = np.unique(dataset.sols.copy(), axis=0) # remove duplicate

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu").numpy()
        # solve
        if np.random.uniform() <= self.solve_ratio:
            sol, _ = _solve_in_pass(cp, self.optmodel, self.processes, self.pool)
            # add into solpool
            self._update_solution_pool(sol)
        solpool = torch.from_numpy(self.solpool.astype(np.float32)).to(device)
        # get current obj
        obj_cp = torch.einsum("bd,bd->b", pred_cost, true_sol).unsqueeze(1)
        # get obj for solpool
        objpool_cp = torch.einsum("bd,nd->bn", pred_cost, solpool)
        # get loss
        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss, _ = (obj_cp - objpool_cp).max(axis=1)
        elif self.optmodel.modelSense == EPO.MAXIMIZE:
            loss, _ = (objpool_cp - obj_cp).max(axis=1)
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss