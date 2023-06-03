from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
import random
import math
import struct
import numpy as np
class DyanamicDataLoader(Dataset):
    def __init__(self,data_dir,length=None,kind='train',task_loc=None,prompt=None,usage=None):
        self.data_dir=data_dir
        self.kind=kind
        self.images,self.label=load_mnist(self.data_dir,self.kind)
        self.length=length
        self.task_loc='task.txt'
        self.index = []

        if task_loc :
            self.task_loc=task_loc

        for i in range(10):
            sub_index = []
            self.index.append(sub_index)
        for i in range(len(self.label)):
            self.index[self.label[i]].append(i)
        self.task_list=[]
        self.usage=[]
        if prompt==None:
            with open(self.task_loc,'r') as f:
                for line in f.readlines():
                    curLine=line.strip()
                    self.task_list.append(curLine)
        else:
            self.task_list.append(prompt)

        self.transform = transforms.Normalize((0.1307,), (0.3081,))

        if usage==None:

            self.usage=[i for i in range(10,70)]

        else:
            self.usage.append(usage)

    def __getitem__(self, idx):
        task_func=random.choice(self.task_list)

        prompt,num,label=eval(task_func)()

        usage=random.choice(self.usage)

        position=position_encoding(usage,20)

        prompt=torch.concat([prompt,position])

        label=torch.argmax(label).item()

        piece0 = self.images[random.choice(self.index[num[0]])].reshape(28, 28)
        piece1 = self.images[random.choice(self.index[num[1]])].reshape(28, 28)
        piece2 = self.images[random.choice(self.index[num[2]])].reshape(28, 28)
        piece3 = self.images[random.choice(self.index[num[3]])].reshape(28, 28)

        n1 = np.concatenate([piece0, piece1], 1)
        n2 = np.concatenate([piece2, piece3], 1)
        np_img = np.concatenate([n1, n2], 0)

        img_tensor=torch.from_numpy(np_img)
        img_tensor=img_tensor.float()
        img_tensor=img_tensor.unsqueeze(0)


        img_tensor=self.transform(img_tensor)

        return prompt,img_tensor,label,usage

    def __len__(self):
        if self.length:
            return self.length
        else:
            if self.kind=='train':
                return 200000
            else:
                return 100000


def position_encoding(position, d_model):
    position = torch.tensor(position, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    encodings = torch.zeros(d_model)
    encodings[0::2] = torch.sin(position * div_term)
    encodings[1::2] = torch.cos(position * div_term)
    return encodings

def load_mnist(path,kind='train'):
    labels_path=os.path.join(path,'%s-labels-idx1-ubyte'%kind)
    images_path=os.path.join(path,'%s-images-idx3-ubyte'%kind)

    with open(labels_path,'rb') as lbpath:
        magic,n =struct.unpack('>II',lbpath.read(8))
        labels=np.fromfile(lbpath,dtype=np.uint8)


    with open(images_path, 'rb') as imgpath:
        magic,num,rows,cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels),784)

    return images,labels


def Is_there_a_number_0():
    label=random.choice([True,False])
    prompt=torch.tensor([1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
    num=[]
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 0
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([1,2,3,4,5,6,7,8,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num,label
    else:
        for k in range(4):
            rest=random.choice([1,2,3,4,5,6,7,8,9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt,num,label

def Is_there_a_number_1():
    label = random.choice([True, False])
    prompt = torch.tensor([1,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0])
    num = []
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 1
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num, label
    else:
        for k in range(4):
            rest = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_a_number_2():
    label = random.choice([True, False])
    prompt = torch.tensor([1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0])
    num = []
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 2
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 1, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num, label
    else:
        for k in range(4):
            rest = random.choice([0, 1, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_a_number_3():
    label = random.choice([True, False])
    prompt = torch.tensor([1,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0])
    num = []
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 3
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 1, 2, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num, label
    else:
        for k in range(4):
            rest = random.choice([0, 1, 2, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_a_number_4():
    label = random.choice([True, False])
    prompt = torch.tensor([1,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0])
    num = []
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 4
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 1, 2, 3, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num, label
    else:
        for k in range(4):
            rest = random.choice([0, 1, 2, 3, 5, 6, 7, 8, 9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_a_number_5():
    label = random.choice([True, False])
    prompt = torch.tensor([1,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0])
    num = []
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 5
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num, label
    else:
        for k in range(4):
            rest = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_a_number_6():
    label = random.choice([True, False])
    prompt = torch.tensor([1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0])
    num = []
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 6
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 1, 2, 3, 4, 5, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num, label
    else:
        for k in range(4):
            rest = random.choice([0, 1, 2, 3, 4, 5, 7, 8, 9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_a_number_7():
    label = random.choice([True, False])
    prompt = torch.tensor([1,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0])
    num = []
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 7
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 1, 2, 3, 4, 5, 6, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num, label
    else:
        for k in range(4):
            rest = random.choice([0, 1, 2, 3, 4, 5, 6, 8, 9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_a_number_8():
    label = random.choice([True, False])
    prompt = torch.tensor([1,0,1,1,1,1,0,0,0,0,0,0,0,0,1,0])
    num = []
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 8
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 1, 2, 3,4, 5, 6, 7, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num, label
    else:
        for k in range(4):
            rest = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_a_number_9():
    label = random.choice([True, False])
    prompt = torch.tensor([1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1])
    num = []
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 9
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num, label
    else:
        for k in range(4):
            rest = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_one_number_0():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0])
    num=[]
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 0
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([1,2,3,4,5,6,7,8,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num,label
    else:

        target_number = random.choice([0,2,3,4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 0
            num.append(target)
        for k in range(rest_number):
            rest=random.choice([1,2,3,4,5,6,7,8,9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt,num,label

def Is_there_only_two_number_0():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0])
    num=[]
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 0
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([1,2,3,4,5,6,7,8,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num,label
    else:

        target_number = random.choice([0,1,3,4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 0
            num.append(target)
        for k in range(rest_number):
            rest=random.choice([1,2,3,4,5,6,7,8,9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt,num,label

def Is_there_only_three_number_0():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0])
    num=[]
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 0
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([1,2,3,4,5,6,7,8,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num,label
    else:

        target_number = random.choice([0,1,2,4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 0
            num.append(target)
        for k in range(rest_number):
            rest=random.choice([1,2,3,4,5,6,7,8,9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt,num,label

def Is_there_only_four_number_0():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0])
    num=[]
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 0
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([1,2,3,4,5,6,7,8,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num,label
    else:

        target_number = random.choice([0,1,2,3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 0
            num.append(target)
        for k in range(rest_number):
            rest=random.choice([1,2,3,4,5,6,7,8,9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt,num,label

def Is_there_only_one_number_1():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 1
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 1
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_two_number_1():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 1
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 1
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_three_number_1():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 1
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 1
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_number_1():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 1
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 1
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0, 2, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_one_number_2():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 2
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 2
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_two_number_2():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 2
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 2
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_three_number_2():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 2
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 2
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_number_2():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 2
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 2
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 3, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_one_number_3():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 3
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 3
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_two_number_3():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 3
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 3
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_three_number_3():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 3
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 3
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_number_3():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0])
    num = []
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 3
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 4, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 3
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 4, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_one_number_4():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0])
    num = []
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 4
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 4
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_two_number_4():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0])
    num = []
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 4
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 4
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_three_number_4():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0])
    num = []
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 4
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 4
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_number_4():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0])
    num = []
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 4
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 5, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 4
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 5, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_one_number_5():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0])
    num = []
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 5
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 5
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_two_number_5():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0])
    num = []
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 5
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 5
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_three_number_5():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0])
    num = []
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 5
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 5
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_number_5():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0])
    num = []
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 5
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 6, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:

        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 5
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 6, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_one_number_6():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0])
    num = []
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 6
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 6
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_two_number_6():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0])
    num = []
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 6
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 6
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_three_number_6():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0])
    num = []
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 6
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 6
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_number_6():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0])
    num = []
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 6
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 7, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 6
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 7, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_one_number_7():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0])
    num = []
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 7
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 7
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_two_number_7():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0])
    num = []
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 7
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 7
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_three_number_7():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0])
    num = []
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 7
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 7
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_number_7():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0])
    num = []
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 7
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 8, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 7
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 8, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_one_number_8():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0])
    num = []
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 8
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 8
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_two_number_8():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0])
    num = []
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 8
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 8
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_three_number_8():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0])
    num = []
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 8
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 8
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_number_8():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0])
    num = []
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 8
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 8
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_one_number_9():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1])
    num = []
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 9
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 9
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 8])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_two_number_9():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1])
    num = []
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 9
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 9
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 8])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_three_number_9():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1])
    num = []
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 9
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 9
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 8])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_number_9():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1])
    num = []
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 9
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = 9
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1, 2, 3, 4, 5, 6, 7, 8])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_odd_numbers():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1])
    num=[]
    if label:
        for k in range(4):
            target = random.choice([1,3,5,7,9])
            num.append(target)
        random.shuffle(num)
        label = torch.tensor([1,0])
        return prompt, num, label

    else:
        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number
        for k in range(target_number):
            target = random.choice([1,3,5,7,9])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,2,4,6,8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt ,num, label

def Is_there_only_three_odd_numbers():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1])
    num=[]
    if label:
        for k in range(3):
            target = random.choice([1,3,5,7,9])
            num.append(target)
        for k in range(1):
            rest = random.choice([0,2,4,6,8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1,0])
        return prompt, num, label

    else:
        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number
        for k in range(target_number):
            target = random.choice([1,3,5,7,9])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,2,4,6,8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt ,num, label

def Is_there_only_two_odd_numbers():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1])
    num=[]
    if label:
        for k in range(2):
            target = random.choice([1,3,5,7,9])
            num.append(target)
        for k in range(2):
            rest = random.choice([0,2,4,6,8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1,0])
        return prompt, num, label

    else:
        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number
        for k in range(target_number):
            target = random.choice([1,3,5,7,9])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,2,4,6,8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt ,num, label

def Is_there_only_one_odd_number():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,1,0,0,0,0,1,0,1,0,1,0,1,0,1])
    num=[]
    if label:
        for k in range(1):
            target = random.choice([1,3,5,7,9])
            num.append(target)
        for k in range(3):
            rest = random.choice([0,2,4,6,8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1,0])
        return prompt, num, label

    else:
        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number
        for k in range(target_number):
            target = random.choice([1,3,5,7,9])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,2,4,6,8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt ,num, label

def Is_there_only_one_even_number():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,1,0,0,0,1,0,1,0,1,0,1,0,1,0])
    num=[]
    if label:
        for k in range(1):
            target = random.choice([0,2,4,6,8])
            num.append(target)
        for k in range(3):
            rest = random.choice([1,3,5,7,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1,0])
        return prompt, num, label

    else:
        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number
        for k in range(target_number):
            target = random.choice([0,2,4,6,8])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([1,3,5,7,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt ,num, label

def Is_there_only_two_even_numbers():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0])
    num=[]
    if label:
        for k in range(2):
            target = random.choice([0,2,4,6,8])
            num.append(target)
        for k in range(2):
            rest = random.choice([1,3,5,7,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1,0])
        return prompt, num, label

    else:
        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number
        for k in range(target_number):
            target = random.choice([0,2,4,6,8])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([1,3,5,7,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt ,num, label

def Is_there_only_three_even_numbers():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0])
    num=[]
    if label:
        for k in range(3):
            target = random.choice([0,2,4,6,8])
            num.append(target)
        for k in range(1):
            rest = random.choice([1,3,5,7,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1,0])
        return prompt, num, label

    else:
        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number
        for k in range(target_number):
            target = random.choice([0,2,4,6,8])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([1,3,5,7,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt ,num, label

def Is_there_only_four_even_numbers():
    label=random.choice([True,False])
    prompt=torch.tensor([0,1,0,0,0,1,1,0,1,0,1,0,1,0,1,0])
    num=[]
    if label:
        for k in range(4):
            target = random.choice([0,2,4,6,8])
            num.append(target)
        for k in range(0):
            rest = random.choice([1,3,5,7,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1,0])
        return prompt, num, label

    else:
        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number
        for k in range(target_number):
            target = random.choice([0,2,4,6,8])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([1,3,5,7,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt ,num, label

def Is_there_an_odd_number():
    label=random.choice([True,False])
    prompt=torch.tensor([1,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1])
    num=[]
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([1,3,5,7,9])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,2,4,6,8])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num,label
    else:
        for k in range(4):
            rest=random.choice([0,2,4,6,8])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt,num,label

def Is_there_an_even_number():
    label=random.choice([True,False])
    prompt=torch.tensor([1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,0])
    num=[]
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([0,2,4,6,8])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([1,3,5,7,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num,label
    else:
        for k in range(4):
            rest=random.choice([1,3,5,7,9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt,num,label

def Is_there_a_prime_number():
    label=random.choice([True,False])
    prompt=torch.tensor([1,0,1,1,1,1,0,0,1,0,0,1,0,1,0,0])
    num=[]
    if label:
        target_number = random.choice([1, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([2,3,5,7])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1,4,6,8,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt,num,label
    else:
        for k in range(4):
            rest=random.choice([0,1,4,6,8,9])
            num.append(rest)

        label = torch.tensor([0, 1])
        return prompt,num,label

def Is_there_only_one_prime_numbers():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,1,0,0,0,0,0,1,0,0,1,0,1,0,0])
    num = []
    if label:
        target_number = 1
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([2,3,5,7])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1,4,6,8,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 2, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([2,3,5,7])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1,4,6,8,9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_two_prime_numbers():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0])
    num = []
    if label:
        target_number = 2
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([2,3,5,7])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1,4,6,8,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 3, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([2,3,5,7])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1,4,6,8,9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_three_prime_numbers():
    label = random.choice([True, False])
    prompt = torch.tensor([0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0])
    num = []
    if label:
        target_number = 3
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([2,3,5,7])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1,4,6,8,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 4])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([2,3,5,7])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1,4,6,8,9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label

def Is_there_only_four_prime_numbers():
    label = random.choice([True, False])
    prompt =  torch.tensor([0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,0])
    num = []
    if label:
        target_number = 4
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([2,3,5,7])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1,4,6,8,9])
            num.append(rest)
        random.shuffle(num)
        label = torch.tensor([1, 0])
        return prompt, num, label
    else:
        target_number = random.choice([0, 1, 2, 3])
        rest_number = 4 - target_number

        for k in range(target_number):
            target = random.choice([2,3,5,7])
            num.append(target)
        for k in range(rest_number):
            rest = random.choice([0,1,4,6,8,9])
            num.append(rest)

        random.shuffle(num)
        label = torch.tensor([0, 1])
        return prompt, num, label












