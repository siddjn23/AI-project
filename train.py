import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from PIL import Image

def DataSet_Values(args):
    
    data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
   Data_Transforms=transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406], 
                                                            [0.229,0.224,0.225])])

Valid_Transforms=transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406], 
                                                           [0.229,0.224,0.225])])

Test_Transforms=transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406], 
                                                           [0.229,0.224,0.225])])

#TODO: Load the datasets with ImageFolder
ImageDatasets=dict()
ImageDatasets['train']=datasets.ImageFolder(train_dir,transform=Data_Transforms)
ImageDatasets['valid']=datasets.ImageFolder(valid_dir,transform=Valid_Transforms)
ImageDatasets['test']=datasets.ImageFolder(test_dir,transform=Test_Transforms)


#TODO: Using the image datasets and the trainforms, define the DataLoaders
DataLoaders=dict()
DataLoaders['train']=torch.utils.data.DataLoader(ImageDatasets['train'],batch_size=64,shuffle=True)
DataLoaders['valid']=torch.utils.data.DataLoader(ImageDatasets['valid'],batch_size=32)
DataLoaders['test']=torch.utils.data.DataLoader(ImageDatasets['test'],batch_size=32)
    return dataloaders,image_datasets

def Training_Of_Model(model,Criterion,Optimizer,Epochs):
    SteProbability=40
    RunningLoss=0
    No_of_Steps=0
    if System_Cuda:
        print('GPU UNDER TRAINING')
        model.cuda()
    elif System_Cuda==False:
        print('GPU UNDER PROCESSING')
    else:
        print('CPU UNDER TESTING')
    for e in range(Epochs):  
        model.train()
        for Images,Labels in iter(DataLoaders['train']):
            No_of_Steps=No_of_Steps+1
            Images=Variable(Images)
            Labels=Variable(Labels)
            if System_Cuda:
                Images=Images.cuda()
                Labels=Labels.cuda()
            Optimizer.zero_grad()
            Output=model.forward(Images)
            Loss=Criterion(Output,Labels)
            Loss.backward()
            Optimizer.step()
            RunningLoss=RunningLoss+Loss.data.item()
            
            if No_of_Steps%SteProbability==0:
                model.eval()
                Accuracy=0
                ValidLoss=0

                for Images,Labels in iter(DataLoaders['valid']):
                    Images=Variable(Images)
                    Labels=Variable(Labels)
                    if System_Cuda:
                        Images=Images.cuda()
                        Labels=Labels.cuda()
                    with torch.no_grad():
                        Output=model.forward(Images)
                        ValidLoss += Criterion(Output,Labels).data.item()
                        Probability=torch.exp(Output).data
                        Equality=(Labels.data==Probability.max(1)[1])
                        Accuracy += Equality.type_as(torch.FloatTensor()).mean()
        
                print("Epoch: {}/{}".format(e+1, Epochs), '\n' ,
                      
                 "Training_Loss_Value: {:.3f}".format(RunningLoss/SteProbability),
                    '\n',
                  "Valid_Loss_Value: {:.3f}".format(ValidLoss/len(DataLoaders['valid'])),
                      '\n',
                  "Valid_Accuracy_Value: {:.3f}".format(Accuracy/len(DataLoaders['valid'])))
                print('\n')
                print('\n')
                RunningLoss=0
                model.train()
            
    print('{} Epochs COMPLETE. MODEL TRAINED.'.format(Epochs))
Training_Of_Model(model,Criterion,Optimizer,5)

    return model

def Model_Architecture(args):
    dataloaders,validloader=DataSet_Values(args)
    if args.arch=='vgg':
        model=models.vgg16(pretrained=True)
        Initial_Input = model.classifier[0].in_features
    elif args.arch=='densenet':
        model=models.densenet121(pretrained=True)
        Initial_Input = model.classifier.in_features
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(Initial_Input,512)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('hidden', nn.Linear(512,args.hidden_units)),                       
                              ('fc2', nn.Linear(args.hidden_units,102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
    model.classifier = classifier

    if args.gpu:
        if torch.cuda.is_available():
            model = model.cuda()
            print ("Inside GPU: "+ str(torch.cuda.is_available()))
        else:
            print("Inside CPU")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=args.lr)
    model = Train(args,model,criterion,optimizer,args.epochs)

    model.class_to_idx=dataloadrs['train'].dataset.class_to_idx
    model.epochs=args.epochs
    checkpoint={'input_size':[3,224,224],
                  'batch_size':dataloaders['train'].batch_size,
                  'output_size':102,
                  'arch':args.arch,
                  'state_dict':model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx':model.class_to_idx,
                  'epoch':model.epochs}
    torch.save(checkpoint,args.saved_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flower Classifcation')
    parser.add_argument('--gpu', type=bool, default=False, help='CHECK GPU AVAILIBILTY')
    parser.add_argument('--arch', type=str, default='densenet', help='CHECK ARCHITECTURE OF PREDIFINED NETWORKS [available: densenet, vgg]', required=True)
    parser.add_argument('--lr', type=float, default=0.001, help='LEARNING RATE')
    parser.add_argument('--hidden_units', type=int, default=100, help='HIDDEN UNITS FOR FC LAYERS')
    parser.add_argument('--epochs', type=int, default=15, help='EPOCHES')
    parser.add_argument('--data_dir', type=str, default='flowers', help='DATASET DIRECTORY')
    parser.add_argument('--saved_model' , type=str, default='my_checkpoint_cmd.pth', help='PATH OF SAVED MODEL')
    args = parser.parse_args()

    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    Model_Architecture(args)
