import math
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from sklearn import preprocessing
from numpy.linalg import norm

# Below is for graph learning part
from torch_geometric.nn.conv import MessagePassing
from typing import Optional
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum


device = torch.device("cuda:1")


# def set_device(device_name):
#     device = torch.device(device_name)


# loss for binary classification
def lr_loss(w, X, y, lam):
    '''
    input: 
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
    return:
        averaged training loss with L2 regularization
    '''
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2


# evaluate function for binary classification
def lr_eval(w, X, y):
    '''
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
    return:
        prediction accuracy
    '''
    return X.mv(w).sign().eq(y).float().mean()


# gradient of loss wrt w for binary classification
def lr_grad(w, X, y, lam):
    '''
    The gradient here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
    return:
        gradient: (d,)
    '''
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z-1) * y) + lam * X.size(0) * w


# hessian of loss wrt w for binary classification
def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    '''
    The hessian here is computed wrt sum.
    input:
        w: (d,)
        X: (n,d)
        y: (n,)
        lambda: scalar
        batch_size: int
    return:
        hessian: (d,d)
    '''
    z = torch.sigmoid(y * X.mv(w))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()


# training iteration for binary classification
def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-32, verbose=False, opt_choice='LBFGS', lr=0.01, wd=0, X_val=None, y_val=None):
    '''
        b is the noise here. It is either pre-computed for worst-case, or pre-defined.
    '''
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)
    
    if opt_choice == 'LBFGS':
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == 'Adam':
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise("Error: Not supported optimizer.")
    
    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()
        
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        
        if opt_choice == 'LBFGS':
            optimizer.step(closure)
        elif opt_choice == 'Adam':
            optimizer.step()
        else:
            raise("Error: Not supported optimizer.")
        
        # If we want to control the norm of w_best, we should keep the last w instead of the one with
        # the highest val acc
        if X_val is not None:
            val_acc = lr_eval(w, X_val, y_val)
            if verbose:
                print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()

    if w_best is None:
        raise("Error: Training procedure failed")
    return w_best


# aggregated loss for multiclass classification
def ovr_lr_loss(w, X, y, lam, weight=None):
    '''
     input:
        w: (d,c)
        X: (n,d)
        y: (n,c), one-hot
        lambda: scalar
        weight: (c,) / None
    return:
        loss: scalar
    '''
    z = batch_multiply(X, w) * y
    if weight is None:
        return -F.logsigmoid(z).mean(0).sum() + lam * w.pow(2).sum() / 2
    else:
        return -F.logsigmoid(z).mul_(weight).sum() + lam * w.pow(2).sum() / 2


def ovr_lr_eval(w, X, y):
    '''
    input:
        w: (d,c)
        X: (n,d)
        y: (n,), NOT one-hot
    return:
        loss: scalar
    '''
    pred = X.mm(w).max(1)[1]
    return pred.eq(y).float().mean()


def ovr_lr_optimize(X, y, lam, weight=None, b=None, num_steps=100, tol=1e-32, verbose=False, opt_choice='LBFGS', lr=0.01, wd=0, X_val=None, y_val=None):
    '''
    y: (n_train, c). one-hot
    y_val: (n_val,) NOT one-hot
    '''
    # We use random initialization as in common DL literature.
    # w = torch.zeros(X.size(1), y.size(1)).float()
    # init.kaiming_uniform_(w, a=math.sqrt(5))
    # w = torch.autograd.Variable(w.to(device), requires_grad=True)
    # zero initialization
    w = torch.autograd.Variable(torch.zeros(X.size(1), y.size(1)).float().to(device), requires_grad=True)

    def closure():
        if b is None:
            return ovr_lr_loss(w, X, y, lam, weight)
        else:
            return ovr_lr_loss(w, X, y, lam, weight) + (b * w).sum() / X.size(0)
    
    if opt_choice == 'LBFGS':
        optimizer = optim.LBFGS([w], lr=lr, tolerance_grad=tol, tolerance_change=1e-32)
    elif opt_choice == 'Adam':
        optimizer = optim.Adam([w], lr=lr, weight_decay=wd)
    else:
        raise("Error: Not supported optimizer.")
    
    best_val_acc = 0
    w_best = None
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = ovr_lr_loss(w, X, y, lam, weight)
        if b is not None:
            if weight is None:
                loss += (b * w).sum() / X.size(0)
            else:
                loss += ((b * w).sum(0) * weight.max(0)[0]).sum()
        loss.backward()
        
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        
        if opt_choice == 'LBFGS':
            optimizer.step(closure)
        elif opt_choice == 'Adam':
            optimizer.step()
        else:
            raise("Error: Not supported optimizer.")
        
        if X_val is not None:
            val_acc = ovr_lr_eval(w, X_val, y_val)
            if verbose:
                print('Val accuracy = %.4f' % val_acc, 'Best Val acc = %.4f' % best_val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                w_best = w.clone().detach()
        else:
            w_best = w.clone().detach()
    
    if w_best is None:
        raise("Error: Training procedure failed")
    return w_best


def batch_multiply(A, B, batch_size=500000):
    if A.is_cuda:
        if len(B.size()) == 1:
            return A.mv(B)
        else:
            return A.mm(B)
    else:
        out = []
        num_batch = int(math.ceil(A.size(0) / float(batch_size)))
        with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i+1) * batch_size, A.size(0))
                A_sub = A[lower:upper]
                A_sub = A_sub.to(device)
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
        return torch.cat(out, dim=0).to(device)


# using power iteration to find the maximum eigenvalue
def sqrt_spectral_norm(A, num_iters=100):
    '''
    return:
        sqrt of maximum eigenvalue/spectral norm
    '''
    x = torch.randn(A.size(0)).float().to(device)
    for i in range(num_iters):
        x = A.mv(x)
        x_norm = x.norm()
        x /= x_norm
    max_lam = torch.dot(x, A.mv(x)) / torch.dot(x, x)
    return math.sqrt(max_lam)


# prepare P matrix in PyG format
def get_propagation(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None, alpha=0.5):
    """
    return:
        P = D^{-\alpha}AD^{-(1-alpha)}.
    """
    fill_value = 2. if improved else 1.
    assert (0 <= alpha) and (alpha <= 1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_left = deg.pow(-alpha)
    deg_inv_right = deg.pow(alpha-1)
    deg_inv_left.masked_fill_(deg_inv_left == float('inf'), 0)
    deg_inv_right.masked_fill_(deg_inv_right == float('inf'), 0)

    return edge_index, deg_inv_left[row] * edge_weight * deg_inv_right[col]


class MyGraphConv(MessagePassing):
    """
    Use customized propagation matrix. Just PX (or PD^{-1}X), no linear layer yet.
    """
    _cached_x: Optional[Tensor]

    def __init__(self, K: int = 1,
                 add_self_loops: bool = True,
                 alpha=0.5, XdegNorm=False, GPR=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.K = K
        self.add_self_loops = add_self_loops
        self.alpha = alpha
        self.XdegNorm = XdegNorm
        self.GPR = GPR
        self._cached_x = None # Not used
        self.reset_parameters()

    def reset_parameters(self):
        self._cached_x = None # Not used

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = get_propagation(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype, alpha=self.alpha)
        elif isinstance(edge_index, SparseTensor):
            edge_index = get_propagation(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype, alpha=self.alpha)
        
        if self.XdegNorm:
            # X <-- D^{-1}X, our degree normalization trick
            num_nodes = maybe_num_nodes(edge_index, None)
            row, col = edge_index[0], edge_index[1]
            deg = degree(row).unsqueeze(-1)
            
            deg_inv = deg.pow(-1)
            deg_inv = deg_inv.masked_fill_(deg_inv == float('inf'), 0) 
        
        if self.GPR:
            xs = []
            xs.append(x)
            if self.XdegNorm:
                x = deg_inv * x # X <-- D^{-1}X
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                xs.append(x)
            return torch.cat(xs, dim=1) / (self.K + 1)
        else:
            if self.XdegNorm:
                x = deg_inv * x # X <-- D^{-1}X
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')


# K = X^T * X for fast computation of spectral norm
def get_K_matrix(X):
    K = X.t().mm(X)
    return K


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, test_lb=1000, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    if Flag == 0:
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
    else:
        all_index = torch.randperm(data.y.shape[0])
        data.val_mask = index_to_mask(all_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(all_index[val_lb: (val_lb+test_lb)], size=data.num_nodes)
        data.train_mask = index_to_mask(all_index[(val_lb+test_lb):], size=data.num_nodes)
    return data


def get_balance_train_mask(y_train, num_classes):
    """
    Make the size of each class in training set = the smallest class.
    """
    # Find the smallest class size
    C_size = torch.zeros(num_classes)
    for i in range(num_classes):
        C_size[i] = (y_train == i).sum()
        
    C_size_exceed = C_size - C_size.min()
    
    # For each class, remove nodes such that size = C_min.
    train_balance_mask = torch.ones(y_train.shape[0])
    All_train_id = np.arange(y_train.shape[0])
    for i in range(num_classes):
        if int(C_size_exceed[i])>0:
            pick = np.random.choice(All_train_id[y_train==i],int(C_size_exceed[i]),replace=False)
            train_balance_mask[pick] = 0
    return train_balance_mask.type(torch.BoolTensor)


def preprocess_data(X):
    '''
    input:
        X: (n,d), torch.Tensor
    '''
    X_np = X.cpu().numpy()
    scaler = preprocessing.StandardScaler().fit(X_np)
    X_scaled = scaler.transform(X_np)
    row_norm = norm(X_scaled, axis=1)
    X_scaled = X_scaled / row_norm.max()
    return torch.from_numpy(X_scaled)


def get_worst_Gbound_feature(lam, m, deg_m, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return gamma2 * ((2*c*lam + (c*gamma1+lam*c1)*deg_m) ** 2) / (lam ** 4) / (m-1)


def get_worst_Gbound_edge(lam, m, K, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return 16 * gamma2 * (K**2) * ((c*gamma1+lam*c1) ** 2) / (lam ** 4) / m


def get_worst_Gbound_node(lam, m, K, deg_m, gamma1=0.25, gamma2=0.25, c=1, c1=1):
    return gamma2 * ((2*c*lam + K*(c*gamma1+lam*c1)*(2*deg_m-1)) ** 2) / (lam ** 4) / (m-1)


def get_c(delta):
    return np.sqrt(2*np.log(1.5/delta))


def get_budget(std, eps, c):
    return std * eps / c
