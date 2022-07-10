import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F




class Resnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=1000, t=0, adj_file=None):
        super(Resnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.incha = in_channel
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(7, 7)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, feature, inp):
        feature = self.features(feature)
        #print('fea:', feature.shape)
        feature = self.pooling(feature)
        #print('fea:', feature.shape)
        feature = self.fc1(feature.squeeze())
        feature = self.relu(feature)
        feature = self.fc2(feature)
        return feature

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.fc1.parameters(), 'lr': lr},
            {'params': self.fc2.parameters(), 'lr': lr},
            #{'params': self.gc1.parameters(), 'lr': lr},
            #{'params': self.gc2.parameters(), 'lr': lr},
        ]





def get_cos(fea, x):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    rp = []
    soft = nn.Softmax(dim=0)
    for i in range(fea.size(0)):
        tmp = cos(fea[i].view(1, -1), x)
        tmp = soft(tmp)
        # tmp /= tmp.sum()
        # tmp = torch.sigmoid(tmp)
        rp.append(tmp)
    resp = torch.cat(rp, 0)
    return resp

def wei_init(wei):
    #stdv = 1. / math.sqrt(wei.size(0))
    #wei.data.uniform_(-stdv, stdv)
    return wei




def gcn_resnet101(num_classes, t, pretrained=True, adj_file=None, in_channel=1000):
    model = models.resnet50(pretrained=pretrained)
    return Resnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
