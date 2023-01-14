import torch
import torch.nn as nn
import numpy as np
import networkx as nx

# class TanhNewtonImplicitLayer(nn.Module):
#     def __init__(self, out_features, tol = 1e-4, max_iter=50):
#         super().__init__()
#         self.linear = nn.Linear(out_features, out_features, bias=False)
#         self.tol = tol
#         self.max_iter = max_iter
  
#     def forward(self, x):
#         # Run Newton's method outside of the autograd framework
#         # It's like an oracle that tells you what z^* should be.
#         with torch.no_grad():
#             z = torch.tanh(x)
#             self.iterations = 0
#             while self.iterations < self.max_iter:
#                 z_linear = self.linear(z) + x
#                 g = z - torch.tanh(z_linear)
#                 self.err = torch.norm(g)
#                 if self.err < self.tol:
#                     break

#                 # newton step
#                 J = torch.eye(z.shape[1])[None,:,:] - (1 / torch.cosh(z_linear)**2)[:,:,None]*self.linear.weight[None,:,:]
#                 # z = z - torch.linalg.solve(g[:,:,None], J)[0][:,:,0]
#                 z = z - torch.linalg.solve(J, g[:,:,None])[0][:,:,0]
#                 self.iterations += 1
    
#         # reengage autograd and add the gradient hook
#         z = torch.tanh(self.linear(z) + x)
#         # this tells pytorch how to backprop
#         z.register_hook(lambda grad : torch.solve(grad[:,:,None], J.transpose(1,2))[0][:,:,0])
#         return z

# from torch.autograd import gradcheck

# layer = TanhNewtonImplicitLayer(5, tol=1e-10).double()
# gradcheck(layer, torch.randn(3, 5, requires_grad=True, dtype=torch.double), check_undefined_grad=False)


# def add(a,b):
#     a = a + 1
#     b = b + 2
#     return b

# a, b = 0,0
# b = add(a,b)

# print (a,b)

def generate_knn_graph(num_points=10, k=3):
    points = torch.Tensor(np.random.rand(num_points, 2)) # n x 2
    P = torch.cdist(points, points)      # n x n
    I = torch.topk(P, k, dim=1, largest=False)[1]
    I = I[:, 1:] # remove the node itself

    points = points.numpy()

    pos = {}
    for i, p in enumerate(points):
        pos[i] = tuple(p)

    G = nx.Graph()
    for i in range(len(I)):
        for j in I[i]:
            G.add_edge(i,int(j))
    
    return G, pos, points


def get_adj_mtx(G):
    n = len(G.nodes)
    A = np.zeros((n, n), dtype=np.int8)
    for e in G.edges:
        A[e[0], e[1]] = 1
        A[e[1], e[0]] = 1
    
    return A
        

def generate_dist_graph(num_points=10, thres=0.3):
    points = torch.Tensor(np.random.rand(num_points, 2)) # n x 2
    P     = torch.cdist(points, points)      # n x n
    E = torch.stack(torch.where(P < thres), dim=0)

    points = points.numpy()

    pos = {}
    for i, p in enumerate(points):
        pos[i] = tuple(p)

    G = nx.Graph()
    for e in E.transpose(1,0):
        if e[0] != e[1]:
            G.add_edge(e[0], e[1])
        
    return G, pos, points

G_knn, pos, pts = generate_knn_graph()
# G, pos = generate_dist_graph()

A_knn = get_adj_mtx(G_knn)

print ("DONE")