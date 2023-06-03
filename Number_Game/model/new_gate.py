import torch
import torch.nn as nn
import torch.nn.functional as F


def get_TGNetwork(d_model=36, gate_len=16, select_embed_len=128, kernel_number=512, expert_num=1):
    model = TGNetwork(d_model, gate_len, select_embed_len, kernel_number, expert_num)
    return model


class TGNetwork(nn.Module):
    def __init__(self, d_model, gate_len=0, select_embed_len=0, kernel_number=0, expert_num=0):
        super(TGNetwork, self).__init__()
        self.gate_len = gate_len
        self.select_embed_len = select_embed_len
        self.kernel_number = kernel_number
        self.expert_num = expert_num


        self.TaskLinear = nn.Sequential(
            nn.Linear(d_model, gate_len * 8, bias=False),
            nn.BatchNorm1d(gate_len * 8),
            nn.ReLU(),
            nn.Linear(gate_len * 8, gate_len * 64, bias=False),
            nn.BatchNorm1d(gate_len * 64),
            nn.ReLU(),
            nn.Linear(gate_len * 64, gate_len * 128, bias=False),
            nn.BatchNorm1d(gate_len * 128),
            nn.ReLU(),
            nn.Linear(gate_len * 128, gate_len * select_embed_len, bias=False),
            nn.BatchNorm1d(gate_len * select_embed_len),
            nn.ReLU(),
        )

        self.bn_rep = nn.ModuleList()
        for _ in range(expert_num):
            self.bn_rep.append(nn.ModuleList([
                nn.BatchNorm1d(4 * select_embed_len) for _ in range(self.gate_len)
            ]))
        self.act = nn.ReLU(inplace=True)

        self.expert_linear = nn.ModuleList()
        for _ in range(expert_num):
            self.expert_linear.append(nn.ModuleList([nn.Linear(select_embed_len, 4 * select_embed_len, bias=False),
                                                     nn.Linear(4 * select_embed_len, kernel_number, bias=False)]))

    def forward(self, prompt, ):

        layer_encoding = self.TaskLinear(prompt)


        layer_encoding = layer_encoding.view(-1, self.gate_len, self.select_embed_len)


        layer_encoding_list = [layer_encoding[:, i, :] for i in range(self.gate_len)]

        selection = []

        for expert in range(self.expert_num):
            output = []
            bn = self.bn_rep[expert]
            for layer, bn in zip(layer_encoding_list, bn):
                x_layer = self.expert_linear[expert][0](layer)
                x_layer = bn(x_layer)
                x_layer = self.act(x_layer)
                x_layer = self.expert_linear[expert][1](x_layer)
                output.append(x_layer)
            selection.append(torch.stack(output, dim=1))


        layer_selection = selection[0]


        layer_selection = F.sigmoid(layer_selection)
        layer_selection = torch.clamp(1.2 * layer_selection - 0.1, min=0, max=1)
        discrete_gate = StepFunction.apply(layer_selection)


        layer_selection = discrete_gate

        return layer_selection


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theshold=0.49999):
        return (x > theshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None