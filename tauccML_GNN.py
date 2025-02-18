import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

dtype = torch.float32
torch.set_default_dtype(dtype)
torch.autograd.set_detect_anomaly(True)

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList([GCNConv(input_dim,hidden_dim)])
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x,edge_index)
            x = F.elu(x)
        output = self.fc(x)
        output =  F.softmax(output, dim=1)
        return output

class TwoGNN(nn.Module):
    def __init__(self, input_dimx, input_dimy, num_components, hidden_dim, output_dim, num_layers, lr, exp, data, device):
        super(TwoGNN, self).__init__()
        self.gnnx = GNN(num_components, hidden_dim, output_dim, num_layers).to(device)
        self.gnny = GNN(num_components, hidden_dim, output_dim, num_layers).to(device)
        self.dev = device
        # Partitions (one-hot encoded) and data
        self.data = data.to(dtype).T
        self.col_labels_ = torch.full((input_dimy, output_dim), fill_value=0.0)#, requires_grad=False) 
        self.row_labels_ = torch.full((input_dimx, output_dim), fill_value=0.0)#, requires_grad=False)
        self.best_partion = torch.full((input_dimx, output_dim), fill_value=0.0)#, requires_grad=False)
        for i in range(input_dimx):
            j = torch.randint(0, output_dim, (1,))
            self.row_labels_[i,j] = 1
        for i in range(input_dimy):
            j = torch.randint(0, output_dim, (1,))
            self.col_labels_[i,j] = 1
        self.optimizer = optim.Adam(self.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,exp)

    def loss(self, Px, Py, tauyx = False):
        # self.data (m,n), Px (n,k), Py(m,k)
        total = torch.sum(self.data)                    # T
        m1 = torch.matmul(self.data, Px)                # t_{iv} = \sum_{o_u\in CO_i} d_{uv}
        r = torch.matmul(Py.T,m1)                       # t_{ij}
        r = r / total                                   # r_{ij}
        p = torch.sum(r,axis=1)                         # p_i
        q = torch.sum(r,axis=0)                         # q_j
        r_sq = torch.square(r)                          # r^2

        if tauyx:
            mask = (p != 0)                             # protect against 0 division
            num1 = torch.sum(r_sq.T[:,mask] / p[mask])  # first part of the numerator
            q_sqr = torch.sum(torch.square(q))          # q^2
            num = num1 - q_sqr
            denom = 1 - q_sqr
        else:
            mask = (q != 0)
            num1 = torch.sum(r_sq[:,mask] / q[mask])
            p_sqr = torch.sum(torch.square(p))
            num = num1 - p_sqr
            denom = 1 - p_sqr
        
        return -num / denom #compute tau

    def fit(self, x, edge_index_x, y, edge_index_y, max_epochs, threshold, patience, embedding_size,verbose = True):
        self.train()  # Set the model to training mode
        self.tau_x = []
        self.tau_y = []
        min_loss = 0
        loss = 0
        epoch = 0
        last_improvement = 0
        
        while (loss - min_loss) <= threshold and epoch < max_epochs and (epoch - last_improvement) < patience:
            if loss < min_loss :
                min_loss = loss 
                last_improvement = epoch
                self.best_partion = self.row_labels_.detach().clone()
             
            # Forward pass
            outputx = self.gnnx(x, edge_index_x) # (n,k)
            self.row_labels_ = torch.argmax(outputx, dim=1)
            self.row_labels_ = F.one_hot(self.row_labels_, embedding_size).to(dtype)
            # compute tau
            loss1 = self.loss(outputx, self.col_labels_.to(self.dev), False)
            
            # Other side
            outputy = self.gnny(y, edge_index_y)
            self.col_labels_ = torch.argmax(outputy, dim=1)
            self.col_labels_ = F.one_hot(self.col_labels_, embedding_size).to(dtype)
            # compute tau
            loss2 = self.loss(self.row_labels_.to(self.dev), outputy, True)
                                
            # Joint loss
            loss = loss1 + loss2    
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()

            if epoch < 20 : self.scheduler.step()

            if verbose : print('%d, loss: %.3f' %(epoch + 1, -loss))

            self.tau_x.append(-loss1.item())
            self.tau_y.append(-loss2.item())

            epoch += 1

