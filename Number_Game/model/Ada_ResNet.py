import torch.nn as nn
from .new_gate import get_TGNetwork

class Ada_ResNet(nn.Module):
    def __init__(self,ada_kernel,block,layers,num_classes=2,kernel_num=512):
        self.inplanes=64
        super(Ada_ResNet,self).__init__()
        self.ada_kernel=ada_kernel
        self.conv1=nn.Conv2d(1,self.inplanes,kernel_size=7,stride=2,padding=3,bias=True)
        self.bn1=nn.BatchNorm2d(self.inplanes)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #block1
        self.layer1=self._make_layer(block,kernel_num,layers[0])
        #block2
        self.layer2=self._make_layer(block,kernel_num,layers[1])
        #block3
        self.layer3=self._make_layer(block,kernel_num,layers[2],stride=2)
        #blokc4
        self.layer4=self._make_layer(block,kernel_num,layers[3],stride=2)


        if ada_kernel:
            print("enable kernel selection")
            self.select_module = get_TGNetwork(gate_len=sum(layers)*2)
            #FilterSelectModule(selected_length=sum(layers)*2, kernel_number=kernel_num)
        else :
            print("disable kernel selection")

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,block,channels,blocks,stride=1):
        downsample=None
        if stride != 1 or self.inplanes != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, channels, stride=stride, downsample=downsample))
        # 每个blocks的第一个residual结构保存在layers列表中。
        self.inplanes = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, channels))
            # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return layers

    def forward(self,prompt,x):
        self.kernel_selection=None
        if self.ada_kernel:
            self.kernel_selection=self.select_module(prompt)

        if self.kernel_selection!=None:
            x=self.conv1(x)
            x=self.bn1(x)
            x=self.relu(x)
            x=self.maxpool(x)

            gate_num=0
            for layer in self.layer1:
                x=layer(x,self.kernel_selection[:,gate_num:gate_num+2,:])
                gate_num+=2

            for layer in self.layer2:
                x=layer(x,self.kernel_selection[:,gate_num:gate_num+2,:])
                gate_num+=2

            for layer in self.layer3:
                x=layer(x,self.kernel_selection[:,gate_num:gate_num+2,:])
                gate_num+=2

            for layer in self.layer4:
                x=layer(x,self.kernel_selection[:,gate_num:gate_num+2,:])
                gate_num+=2

            x=self.avgpool(x)
            x=x.view(x.size(0),-1)
            x=self.fc(x)

            return x,self.kernel_selection
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            for layer in self.layer1:
                x = layer(x)

            for layer in self.layer2:
                x = layer(x)
            for layer in self.layer3:
                x = layer(x)
            for layer in self.layer4:
                x = layer(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x,self.kernel_selection

















