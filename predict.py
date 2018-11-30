# Imports here
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

def preprocess_image_(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    Size_image=256
    image.thumbnail(Size_image, Image.ANTIALIAS)
    image=image.crop((128 - 112,128-112,128+112,128 + 112))
    Image = np.array(image)
    Me_Image = Image/255
        
    X_1=Me_Image[:,:,0]
    X_2=Me_Image[:,:,1]
    X_3=Me_Image[:,:,2]
    
    X1=(X_1-0.485)/(0.229) 
    X2=(X_2-0.456)/(0.224)
    X3=(X_3-0.406)/(0.225)
    
    Me_Image[:,:,0]=X1
    Me_Image[:,:,1]=X2
    Me_Image[:,:,2]=X3
    
    Me_Image = np.transpose(Me_Image, (2,0,1))
    return Me_Image


# TODO: load the checkpoint
def load_checkpoint(filepath):

    checkpoint = torch.load(args.saved_model)
    if checkpoint['arch'] == 'vgg':
        model = models.vgg16()  
        initial_input= model.classifier[0].in_features
    elif checkpoint['arch'] == 'densenet':
        model = models.densenet121()
        initial_input= model.classifier.in_features
        
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(initial_input, 512)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('hidden', nn.Linear(512, 100)),                       
                              ('fc2', nn.Linear(100, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    if args.gpu:
        if torch.cuda.is_available():
            model = model.cuda()
            print ("Using GPU")
        else:
            print("Using CPU")
    loaded_model, class_2_idx = load_checkpoint('my_checkpoint.pth')
    idx_2_class = { v : k for k,v in class_2_idx.items()}
    return model, class_2_idx, idx_2_class

def predict(args, image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    Me_Image = torch.FloatTensor([process_image(Image.open(image_path))])
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    if args.gpu and torch.cuda.is_available():
        Result = model.forward(Variable(Me_Image.cuda()))
    else:
        Result = model.forward(Variable(Me_Image))
    P_S = torch.exp(Result).data.numpy()[0]
    Top_Index = np.argsort(ps)[-topk:][::-1] 
    Top_Class = [idx_to_class[x] for x in Top_Index]
    Probability_dist = P_S[Top_Index]

    print("Probability is --> {} \n and class is--> {}".format(Probability_dist, Top_Class))
    return Probability_dist, Top_Class

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Flower Classifcation')
    parser.add_argument('--gpu', type=bool, default=False, help='CHECK GPU AVAILIBILTY')
    parser.add_argument('--image_path', type=str, help='IMAGE PATH')
    parser.add_argument('--hidden_units', type=int, default=100, help='HIDDEN UNITS FOR FC LAYERS')
    parser.add_argument('--saved_model' , type=str, default='my_checkpoint_cmd.pth', help='PATH OF SAVED MODEL')
    parser.add_argument('--mapper_json' , type=str, default='cat_to_name.json', help='PATH OF MAPPING ARCHITECTURE FROM COTEGORY TO NAME')
    parser.add_argument('--topk', type=int, default=5, help='TOP K PROBABILITIES')
    args = parser.parse_args()

    import json
    with open(args.mapper_json, 'r') as f:
        cat = json.load(f)

    model, class_2_idx, idx_2_class = load_checkpoint(args)
    T_Probability, T_class = predict(args, args.image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=args.topk)
                                              
    print('Predicted Classes is ---> ', T_Class)
    print ('Class Names:---> ')
    [print(cat[x]) for x in T_Class]
    print('Predicted Probability-----> ', T_Probability)
