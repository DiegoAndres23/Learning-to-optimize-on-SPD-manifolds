import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function


class EigLayerF(Function):
    @staticmethod
    def forward(self,input):

        n=input.shape[0]
        S=torch.zeros(input.shape).cpu()
        U=torch.zeros(input.shape).cpu()

        for i in range(n):
            # value, vector=torch.eig( input[i], eigenvectors=True)
            value, vector = torch.linalg.eig(input[i])

            S[i] = torch.diag(value.real)
            U[i] = vector.real

            
        self.save_for_backward(input, S, U)
        return S,U


    @staticmethod
    def backward(self, grad_S, grad_U):

        input, S, U = self.saved_tensors
        n=input.shape[0]
        dim=input.shape[1]

        grad_input=V( torch.zeros( input.shape ) ).cpu()

        e=torch.eye(dim).cpu()

        P_i=torch.matmul(S,torch.ones(dim,dim).cpu())
        
        P=(P_i-P_i.permute(0,2,1))+e
        epo=(torch.ones(P.shape).cpu())*0.000001
        P=torch.where(P!=0,P,epo)
        P=(1/P)-e
        
        g1= torch.matmul(U.permute(0,2,1),grad_U)
        g1=(g1+g1.permute(0,2,1))/2
        g1=torch.mul(P.permute(0,2,1),g1)
        g1=2* torch.matmul( torch.matmul( U,g1 ), U.permute(0,2,1) )
        g2=torch.matmul( torch.matmul( U, torch.mul(grad_S,e) ), U.permute(0,2,1) )
        grad_input=g1+g2

        return grad_input




class EigLayer(nn.Module):
    def __init__(self):
        super(EigLayer, self).__init__()
    

    def forward(self, input1):
        return EigLayerF().apply(input1)

