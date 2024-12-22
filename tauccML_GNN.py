import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

dtype = torch.float32
torch.set_default_dtype(dtype)
torch.autograd.set_detect_anomaly(True)

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim,   output_dim)
        

    def forward(self, x, adj):
        x = torch.matmul(adj,x)
        x = F.relu(self.linear(x))
        return x

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList([GraphConvolution(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.conv_layers.append(GraphConvolution(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        embeddings = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x,adj)
            embeddings.append(x)
        stacked_embeddings = torch.stack(embeddings, dim=1)  # Stack embeddings along dimension 1
        output = self.fc(x)
        output = nn.functional.softmax(output, dim=1)
        return output, stacked_embeddings

class TwoGNN(nn.Module):
    def __init__(self, input_dimx, input_dimy, hidden_dim, output_dim, num_layers, data, device):
        super(TwoGNN, self).__init__()
        self.gnnx = GNN(input_dimy, hidden_dim, output_dim, num_layers).to(device)
        self.gnny = GNN(input_dimx, hidden_dim, output_dim, num_layers).to(device)
        self.dev = device
        # Partitions (one-hot encoded) and data
        self.data = data.to(torch.float32)
        self.col_labels_ = torch.full((input_dimy, output_dim), fill_value=0.0)#, requires_grad=False) 
        self.row_labels_ = torch.full((input_dimx, output_dim), fill_value=0.0)#, requires_grad=False)
        for i in range(input_dimx):
            j = torch.randint(0, output_dim, (1,))
            self.row_labels_[i,j] = 1
        for i in range(input_dimy):
            j = torch.randint(0, output_dim, (1,))
            self.col_labels_[i,j] = 1
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def loss(self, Px, Py, tauyx = False):
        # self.data (m,n), Px (n,k), Py(m,k)
        total = torch.sum(self.data)                    # T
        m1 = torch.matmul(self.data, Px)                # t_{iv} = \sum_{o_u\in CO_i} d_{uv}
        r = torch.matmul(torch.transpose(Py, 0, 1),m1)  # t_{ij}
        r = r / total                                   # r_{ij}
        p = torch.sum(r,axis=1)                         # p_i
        q = torch.sum(r,axis=0)                         # q_j
        
        r_sq = torch.square(r)                          # r^2

        if(tauyx):
            mask = (p != 0)                             # protect against 0 division
            num1 = torch.sum(r_sq.transpose(0,1)[:,mask] / p[mask]) # first part of the numerator
            q_sqr = torch.sum(torch.square(q))          # q^2
            num = num1 - q_sqr
            denom = 1 - q_sqr
        else:
            mask = (q != 0)
            num1 = torch.sum(r_sq[:,mask] / q[mask])
            p_sqr = torch.sum(torch.square(p))
            num = num1 - p_sqr
            denom = 1 - p_sqr
        #print("loss",num/denom)
        return -num / denom #compute tau

    def fit(self, x, adjx, y, adjy, epochs, dim):
        self.train()  # Set the model to training mode
        for epoch in range(epochs):
            running_loss = 0.0
            if(True):
                # Forward pass
                outputx, _ = self.gnnx(x, adjx) # (n,k)
                self.row_labels_ = torch.argmax(outputx, dim=1)
                self.row_labels_ = F.one_hot(self.row_labels_, dim).to(torch.float32)
                # compute tau
                loss1 = self.loss(outputx, self.col_labels_.to(self.dev), False)
                
                # Other side
                outputy, _ = self.gnny(y, adjy)
                self.col_labels_ = torch.argmax(outputy, dim=1)
                self.col_labels_ = F.one_hot(self.col_labels_, dim).to(torch.float32)
                # compute tau
                loss2 = self.loss(self.row_labels_.to(self.dev), outputy, True)
                                    
                # Joint loss
                loss = loss1 + loss2    
                self.optimizer.zero_grad()
                
                loss.abs().backward()
                self.optimizer.step()
                running_loss += loss.item()
                #if  % 100 == 99:  # Print every 100 mini-batches
                #print('%d, loss: %.3f' %(epoch + 1, -loss))
                running_loss = 0.0

############################################"
##############################################"


    
def test_loss(device):
    n = 5
    p = 4
    hidden_size = 3
    embedding_size = 3
    d = [[3,4,1,1],[5,3,0,2],[6,4,1,0],[0,1,7,7],[1,0,6,8]]
    d = np.array(d)
    d = torch.from_numpy(d).to(device)
    outputx = [[1,0],[1,0],[0,1],[0,1]]
    outputx = np.array(outputx)
    outputy = [[1,0],[1,0],[0,1],[0,1],[0,1]]
    outputy = np.array(outputy)
    outputx = torch.from_numpy(outputx).to(device)
    outputy = torch.from_numpy(outputy).to(device)
    networkpx = NetworkP(n, hidden_size, embedding_size, device)
    networkpy = NetworkP(p, hidden_size, embedding_size, device)
    model = Networks(networkpx, networkpy, 5, 4, 2, d, device)

    one_hot_col = outputx #F.one_hot(outputx, 2)
    one_hot_row = outputy #F.one_hot(outputy, 2)
    loss1 = model.loss1(one_hot_col, one_hot_row, True)
    print("tau yx",-loss1)
    loss2 = model.loss1(one_hot_col, one_hot_row, False)
    print("tau xy",-loss2)
    #one_hot = F.one_hot(outputy, num_classes=dim)
    #print(model.loss1(outputx, outputy, True))
