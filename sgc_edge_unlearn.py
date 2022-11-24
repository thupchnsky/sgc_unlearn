from __future__ import print_function
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import os
from sklearn.linear_model import LogisticRegression

# Below is for graph learning part
from torch_geometric.nn.conv import MessagePassing
from typing import Optional

from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp

from torch.nn import init
from utils import *

from sklearn import preprocessing
from numpy.linalg import norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a removal-enabled linear model [edge]')
    parser.add_argument('--data_dir', type=str, default='./PyG_datasets', help='data directory')
    parser.add_argument('--result_dir', type=str, default='result', help='directory for saving results')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--lam', type=float, default=1e-2, help='L2 regularization')
    parser.add_argument('--std', type=float, default=1e-2, help='standard deviation for objective perturbation')
    parser.add_argument('--num_removes', type=int, default=500, help='number of data points to remove')
    parser.add_argument('--num_steps', type=int, default=100, help='number of optimization steps')
    parser.add_argument('--train_mode', type=str, default='ovr', help='train mode [ovr/binary]')
    parser.add_argument('--train_sep', action='store_true', default=False, help='train binary classifiers separately')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbosity in optimizer')
    # New arguments below
    parser.add_argument('--device', type=int, default=1, help='nonnegative int for cuda id, -1 for cpu')
    parser.add_argument('--prop_step', type=int, default=2, help='number of steps of graph propagation/convolution')
    parser.add_argument('--alpha', type=float, default=0.0, help='we use D^{-a}AD^{-(1-a)} as propagation matrix')
    parser.add_argument('--XdegNorm', type=bool, default=False, help='Apply our degree normaliztion trick')
    parser.add_argument('--add_self_loops', type=bool, default=True, help='Add self loops in propagation matrix')
    parser.add_argument('--optimizer', type=str, default='LBFGS', help='Choice of optimizer. [LBFGS/Adam]')
    parser.add_argument('--lr', type=float, default=1, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay factor for Adam')
    parser.add_argument('--featNorm', type=bool, default=True, help='Row normalize feature to norm 1.')
    parser.add_argument('--GPR', action='store_true', default=False, help='Use GPR model')
    parser.add_argument('--balance_train', action='store_true', default=False,
                        help='Subsample training set to make it balance in class size.')
    parser.add_argument('--Y_binary', type=str, default='0',
                        help='In binary mode, is Y_binary class or Y_binary_1 vs Y_binary_2 (i.e., 0+1).')
    parser.add_argument('--noise_mode', type=str, default='data',
                        help='Data dependent noise or worst case noise [data/worst].')
    parser.add_argument('--removal_mode', type=str, default='edge', help='[feature/edge/node].')
    parser.add_argument('--eps', type=float, default=1.0, help='Eps coefficient for certified removal.')
    parser.add_argument('--delta', type=float, default=1e-4, help='Delta coefficient for certified removal.')
    parser.add_argument('--disp', type=int, default=10, help='Display frequency.')
    parser.add_argument('--trails', type=int, default=10, help='Number of repeated trails.')
    parser.add_argument('--fix_random_seed', action='store_true', default=False,
                        help='Use fixed random seed for removal queue.')
    parser.add_argument('--compare_gnorm', action='store_true', default=False,
                        help='Compute norm of worst case and real gradient each round.')
    parser.add_argument('--compare_retrain', action='store_true', default=False,
                        help='Compare acc with retraining each round.')
    # Use this if turning into .py code
    args = parser.parse_args()

    # Use this if running using notebook
    # args = parser.parse_args([])

    # this script is only for edge removal
    assert args.removal_mode in ['edge']
    # dont compute norm together with retrain
    assert not (args.compare_gnorm and args.compare_retrain)

    if args.device > -1:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    ######
    # Load the data
    print('='*10 + 'Loading data' + '='*10)
    print('Dataset:', args.dataset)
    # read data from PyG datasets (cora, citeseer, pubmed)
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        path = osp.join(args.data_dir, 'data', args.dataset)
        dataset = Planetoid(path, args.dataset, split="full")
        data = dataset[0].to(device)
    elif args.dataset in ['ogbn-arxiv', 'ogbn-products']:
        dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_dir)
        data = dataset[0].to(device)
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.val_mask[split_idx['valid']] = True
        data.test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.test_mask[split_idx['test']] = True
        data.y = data.y.squeeze(-1)
    elif args.dataset in ['computers', 'photo']:
        path = osp.join(args.data_dir, 'data', args.dataset)
        dataset = Amazon(path, args.dataset)
        data = dataset[0]
        data = random_planetoid_splits(data, num_classes=dataset.num_classes, val_lb=500, test_lb=1000, Flag=1).to(device)
    else:
        raise("Error: Not supported dataset yet.")

    # save the degree of each node for later use
    row = data.edge_index[0]
    deg = degree(row)

    # preprocess features
    if args.featNorm:
        X = preprocess_data(data.x).to(device)
    else:
        X = data.x.to(device)
    # save a copy of X for removal
    X_scaled_copy_guo = X.clone().detach().float()

    # process labels
    if args.train_mode == 'binary':
        if '+' in args.Y_binary:
            # two classes are specified
            class1 = int(args.Y_binary.split('+')[0])
            class2 = int(args.Y_binary.split('+')[1])
            Y = data.y.clone().detach().float()
            Y[data.y == class1] = 1
            Y[data.y == class2] = -1
            interested_data_mask = (data.y == class1) + (data.y == class2)
            train_mask = data.train_mask * interested_data_mask
            val_mask = data.val_mask * interested_data_mask
            test_mask = data.test_mask * interested_data_mask
        else:
            # one vs rest
            class1 = int(args.Y_binary)
            Y = data.y.clone().detach().float()
            Y[data.y == class1] = 1
            Y[data.y != class1] = -1
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        y_train, y_val, y_test = Y[train_mask].to(device), Y[val_mask].to(device), Y[test_mask].to(device)
    else:
        # multiclass classification
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        y_train = F.one_hot(data.y[data.train_mask]) * 2 - 1
        y_train = y_train.float().to(device)
        y_val = data.y[data.val_mask].to(device)
        y_test = data.y[data.test_mask].to(device)

    assert args.noise_mode == 'data'

    if args.compare_gnorm:
        # if we want to compare the residual gradient norm of three cases, we should not add noise
        # and make budget very large
        b_std = 0
    else:
        if args.noise_mode == 'data':
            b_std = args.std
        elif args.noise_mode == 'worst':
            b_std = args.std  # change to worst case sigma
        else:
            raise("Error: Not supported noise model.")

    #############
    # initial training with graph
    print('='*10 + 'Training on full dataset with graph' + '='*10)
    start = time.time()
    Propagation = MyGraphConv(K=args.prop_step, add_self_loops=args.add_self_loops,
                              alpha=args.alpha, XdegNorm=args.XdegNorm, GPR=args.GPR).to(device)

    if args.prop_step > 0:
        X = Propagation(X, data.edge_index)

    X = X.float()
    X_train = X[train_mask].to(device)
    X_val = X[val_mask].to(device)
    X_test = X[test_mask].to(device)

    print("Train node:{}, Val node:{}, Test node:{}, Edges:{}, Feature dim:{}".format(X_train.shape[0], X_val.shape[0],
                                                                                      X_test.shape[0],
                                                                                      data.edge_index.shape[1],
                                                                                      X_train.shape[1]))

    # train removal-enabled linear model
    print("With graph, train mode:", args.train_mode, ", optimizer:", args.optimizer)

    weight = None
    # in our case weight should always be None
    assert weight is None
    opt_grad_norm = 0

    if args.train_mode == 'ovr':
        b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
        if args.train_sep:
            # train K binary LR models separately
            w = torch.zeros(b.size()).float().to(device)
            for k in range(y_train.size(1)):
                if weight is None:
                    w[:, k] = lr_optimize(X_train, y_train[:, k], args.lam, b=b[:, k], num_steps=args.num_steps, verbose=args.verbose,
                                          opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                else:
                    w[:, k] = lr_optimize(X_train[weight[:, k].gt(0)], y_train[:, k][weight[:, k].gt(0)], args.lam,
                                          b=b[:, k], num_steps=args.num_steps, verbose=args.verbose, opt_choice=args.optimizer,
                                          lr=args.lr, wd=args.wd)
        else:
            # train K binary LR models jointly
            w = ovr_lr_optimize(X_train, y_train, args.lam, weight, b=b, num_steps=args.num_steps, verbose=args.verbose,
                                opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
        # record the opt_grad_norm
        for k in range(y_train.size(1)):
            opt_grad_norm += lr_grad(w[:, k], X_train, y_train[:, k], args.lam).norm().cpu()
    else:
        b = b_std * torch.randn(X_train.size(1)).float().to(device)
        w = lr_optimize(X_train, y_train, args.lam, b=b, num_steps=args.num_steps, verbose=args.verbose, opt_choice=args.optimizer,
                        lr=args.lr, wd=args.wd)
        opt_grad_norm = lr_grad(w, X_train, y_train, args.lam).norm().cpu()

    print('Time elapsed: %.2fs' % (time.time() - start))
    if args.train_mode == 'ovr':
        print('Val accuracy = %.4f' % ovr_lr_eval(w, X_val, y_val))
        print('Test accuracy = %.4f' % ovr_lr_eval(w, X_test, y_test))
    else:
        print('Val accuracy = %.4f' % lr_eval(w, X_val, y_val))
        print('Test accuracy = %.4f' % lr_eval(w, X_test, y_test))

    ###########
    # budget for removal
    c_val = get_c(args.delta)
    if args.compare_gnorm:
        budget = 1e5
    else:
        if args.train_mode == 'ovr':
            budget = get_budget(b_std, args.eps, c_val) * y_train.size(1)
        else:
            budget = get_budget(b_std, args.eps, c_val)
    gamma = 1/4  # pre-computed for -logsigmoid loss
    print('Budget:', budget)

    ##########
    # our removal
    # grad_norm_approx is the data dependent upper bound of residual gradient norm
    grad_norm_approx = torch.zeros((args.num_removes, args.trails)).float()
    removal_times = torch.zeros((args.num_removes, args.trails)).float()  # record the time of each removal
    acc_removal = torch.zeros((2, args.num_removes, args.trails)).float()  # record the acc after removal, 0 for val, 1 for test
    grad_norm_worst = torch.zeros((args.num_removes, args.trails)).float()  # worst case norm bound
    grad_norm_real = torch.zeros((args.num_removes, args.trails)).float()  # true norm
    # graph retrain
    removal_times_graph_retrain = torch.zeros((args.num_removes, args.trails)).float()
    acc_graph_retrain = torch.zeros((2, args.num_removes, args.trails)).float()

    for trail_iter in range(args.trails):
        print('*'*10, trail_iter, '*'*10)
        if args.fix_random_seed:
            np.random.seed(trail_iter)
        # get a random permutation for edge indices for each trail
        perm = torch.from_numpy(np.random.permutation(data.edge_index.shape[1]))
        # Note that all edges are used in training, so we just need to decide the order to remove edges
        # the number of training samples will always be m
        edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

        X_scaled_copy = X_scaled_copy_guo.clone().detach().float()
        w_approx = w.clone().detach()  # copy the parameters to modify
        X_old = X.clone().detach().to(device)

        num_retrain = 0
        grad_norm_approx_sum = 0
        perm_idx = 0
        # start the removal process
        print('='*10 + 'Testing our edge removal' + '='*10)
        for i in range(args.num_removes):
            # First, check if this is a self-loop or an edge already deleted
            while (data.edge_index[0, perm[perm_idx]] == data.edge_index[1, perm[perm_idx]]) or (not edge_mask[perm[perm_idx]]):
                perm_idx += 1
            edge_mask[perm[perm_idx]] = False
            source_idx = data.edge_index[0, perm[perm_idx]]
            dst_idx = data.edge_index[1, perm[perm_idx]]
            # find the other undirected edge
            rev_edge_idx = torch.logical_and(data.edge_index[0] == dst_idx,
                                             data.edge_index[1] == source_idx).nonzero().squeeze(-1)
            if rev_edge_idx.size(0) > 0:
                edge_mask[rev_edge_idx] = False

            perm_idx += 1
            start = time.time()
            # Get propagated features
            if args.prop_step > 0:
                X_new = Propagation(X_scaled_copy, data.edge_index[:, edge_mask]).to(device)
            else:
                X_new = X_scaled_copy.to(device)

            X_val_new = X_new[val_mask]
            X_test_new = X_new[test_mask]

            K = get_K_matrix(X_new[train_mask]).to(device)
            spec_norm = sqrt_spectral_norm(K)

            if args.train_mode == 'ovr':
                # removal from all one-vs-rest models
                X_rem = X_new[train_mask]
                for k in range(y_train.size(1)):
                    assert weight is None
                    y_rem = y_train[:, k]
                    H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, args.lam)
                    # grad_i is the difference
                    grad_old = lr_grad(w_approx[:, k], X_old[train_mask], y_rem, args.lam)
                    grad_new = lr_grad(w_approx[:, k], X_rem, y_rem, args.lam)
                    grad_i = grad_old - grad_new
                    Delta = H_inv.mv(grad_i)
                    Delta_p = X_rem.mv(Delta)
                    # update w here. If beta exceed the budget, w_approx will be retrained
                    w_approx[:, k] += Delta
                    grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                    if args.compare_gnorm:
                        grad_norm_real[i, trail_iter] += lr_grad(w_approx[:, k], X_rem, y_rem, args.lam).norm().cpu()
                        grad_norm_worst[i, trail_iter] += get_worst_Gbound_edge(args.lam, X_rem.shape[0],
                                                                                args.prop_step)
                # decide after all classes
                if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                    w_approx = ovr_lr_optimize(X_rem, y_train, args.lam, weight, b=b, num_steps=args.num_steps, verbose=args.verbose,
                                               opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                # record acc each round
                acc_removal[0, i, trail_iter] = ovr_lr_eval(w_approx, X_val_new, y_val)
                acc_removal[1, i, trail_iter] = ovr_lr_eval(w_approx, X_test_new, y_test)
            else:
                # removal from a single binary logistic regression model
                X_rem = X_new[train_mask]
                y_rem = y_train
                H_inv = lr_hessian_inv(w_approx, X_rem, y_rem, args.lam)
                # grad_i should be the difference
                grad_old = lr_grad(w_approx, X_old[train_mask], y_train, args.lam)
                grad_new = lr_grad(w_approx, X_rem, y_rem, args.lam)
                grad_i = grad_old - grad_new
                Delta = H_inv.mv(grad_i)
                Delta_p = X_rem.mv(Delta)
                w_approx += Delta
                grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                if args.compare_gnorm:
                    grad_norm_real[i, trail_iter] += lr_grad(w_approx, X_rem, y_rem, args.lam).norm().cpu()
                    grad_norm_worst[i, trail_iter] += get_worst_Gbound_edge(args.lam, X_rem.shape[0], args.prop_step)
                if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = b_std * torch.randn(X_train.size(1)).float().to(device)
                    w_approx = lr_optimize(X_rem, y_rem, args.lam, b=b, num_steps=args.num_steps, verbose=args.verbose, opt_choice=args.optimizer,
                                           lr=args.lr, wd=args.wd)
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                # record acc each round
                acc_removal[0, i, trail_iter] = lr_eval(w_approx, X_val_new, y_val)
                acc_removal[1, i, trail_iter] = lr_eval(w_approx, X_test_new, y_test)

            removal_times[i, trail_iter] = time.time() - start
            # Remember to replace X_old with X_new
            X_old = X_new.clone().detach()
            if i % args.disp == 0:
                print('Iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_times[i, trail_iter], num_retrain))
                print('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i, trail_iter], acc_removal[1, i, trail_iter]))

        #######
        # retrain each round with graph
        if args.compare_retrain:
            edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
            perm_idx = 0
            # start the removal process
            print('='*10 + 'Testing with graph retrain' + '='*10)
            for i in range(args.num_removes):
                # First, check if this is a self-loop or an edge already deleted
                while (data.edge_index[0, perm[perm_idx]] == data.edge_index[1, perm[perm_idx]]) or (not edge_mask[perm[perm_idx]]):
                    perm_idx += 1
                edge_mask[perm[perm_idx]] = False
                source_idx = data.edge_index[0, perm[perm_idx]]
                dst_idx = data.edge_index[1, perm[perm_idx]]
                # find the other undirected edge
                rev_edge_idx = torch.logical_and(data.edge_index[0] == dst_idx,
                                                 data.edge_index[1] == source_idx).nonzero().squeeze(-1)
                if rev_edge_idx.size(0) > 0:
                    edge_mask[rev_edge_idx] = False

                perm_idx += 1
                start = time.time()
                # Get propagated features
                if args.prop_step > 0:
                    X_new = Propagation(X_scaled_copy, data.edge_index[:, edge_mask]).to(device)
                else:
                    X_new = X_scaled_copy.to(device)

                X_val_new = X_new[val_mask]
                X_test_new = X_new[test_mask]

                if args.train_mode == 'ovr':
                    # removal from all one-vs-rest models
                    X_rem = X_new[train_mask]
                    # retrain the model
                    # we do not need to add noise if we are retraining every time
                    # b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                    w_graph_retrain = ovr_lr_optimize(X_rem, y_train, args.lam, weight, b=None, num_steps=args.num_steps, verbose=args.verbose,
                                                      opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    acc_graph_retrain[0, i, trail_iter] = ovr_lr_eval(w_graph_retrain, X_val_new, y_val)
                    acc_graph_retrain[1, i, trail_iter] = ovr_lr_eval(w_graph_retrain, X_test_new, y_test)
                else:
                    # removal from a single binary logistic regression model
                    X_rem = X_new[train_mask]
                    # retrain the model
                    # b = b_std * torch.randn(X_train.size(1)).float().to(device)
                    w_graph_retrain = lr_optimize(X_rem, y_train, args.lam, b=None, num_steps=args.num_steps, verbose=args.verbose,
                                                  opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    acc_graph_retrain[0, i, trail_iter] = lr_eval(w_graph_retrain, X_val_new, y_val)
                    acc_graph_retrain[1, i, trail_iter] = lr_eval(w_graph_retrain, X_test_new, y_test)

                removal_times_graph_retrain[i, trail_iter] = time.time() - start
                if i % args.disp == 0:
                    print('Iteration %d, time = %.2fs, val acc = %.4f, test acc = %.4f' % (i+1, removal_times_graph_retrain[i, trail_iter], acc_graph_retrain[0, i, trail_iter], acc_graph_retrain[1, i, trail_iter]))

    # save all results
    if not osp.exists(args.result_dir):
        os.makedirs(args.result_dir)
    save_path = '%s/%s_std_%.0e_lam_%.0e_nr_%d_K_%d_opt_%s_mode_%s_eps_%.1f_delta_%.0e' % (args.result_dir,
                                                                                           args.dataset, b_std,
                                                                                           args.lam, args.num_removes,
                                                                                           args.prop_step,
                                                                                           args.optimizer,
                                                                                           args.removal_mode,
                                                                                           args.eps, args.delta)
    if args.train_mode == 'binary':
        save_path += '_bin_%s' % args.Y_binary
    if args.GPR:
        save_path += '_gpr'
    if args.compare_gnorm:
        save_path += '_gnorm'
    if args.compare_retrain:
        save_path += '_retrain'
    save_path += '.pth'

    torch.save({'grad_norm_approx': grad_norm_approx, 'removal_times': removal_times, 'acc_removal': acc_removal,
                'grad_norm_worst': grad_norm_worst, 'grad_norm_real': grad_norm_real,
                'removal_times_graph_retrain': removal_times_graph_retrain,
                'acc_graph_retrain': acc_graph_retrain}, save_path)
