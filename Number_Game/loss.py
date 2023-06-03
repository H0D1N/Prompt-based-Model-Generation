import torch.nn as nn
import torch

class sparse_gate_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bass_criterion=nn.CrossEntropyLoss()
        self.gate_loss=nn.MSELoss()
    def forward(self,logists,label,kernel_selection,usage):
        #label=label.float()
        usage=usage/100
        loss = self.bass_criterion(logists, label)

        kernel_selection=kernel_selection.reshape((kernel_selection.size(0),-1))
        selection=torch.norm(kernel_selection,p=1,dim=1)/kernel_selection.size(1)
            #c=kernel_selection.size()[-1]*kernel_selection.size()[-2]*kernel_selection.size()[-3]
            #loss=loss+0.1*torch.pow(((torch.norm(kernel_selection,p=1))/c-0.3),2)
        g_loss=self.gate_loss(selection,usage)
        loss=loss+g_loss
        return loss
