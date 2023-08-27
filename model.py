import torch
from torch import nn
from torchvision import datasets, transforms, models
def create_model():
    model = models.vgg16(pretrained=True)
    hidden_layer=[1000,500]
    input_size=25088
    output_size=102
    classifier1= nn.ModuleList([nn.Linear(input_size, hidden_layer[0])])
    layer_sizes = zip(hidden_layer[:-1], hidden_layer[1:])
    classifier1.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

    output = nn.Linear(hidden_layer[-1], output_size)        
    dropout = nn.Dropout(0.02)
    classifier=nn.ModuleList()
    for each in classifier1:
        classifier.append(each)
        classifier.append(nn.ReLU())
        classifier.append(dropout)
    classifier.append(output)
    classifier.append(nn.LogSoftmax(dim=1))

    classifier=nn.Sequential(*classifier)
    model.classifier = classifier
    return model