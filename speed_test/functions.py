from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torch.nn as nn
import time
from torchvision import datasets, models, transforms

# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# avaiable models in PyTorch: [resnet, alexnet, vgg, squeezenet, densenet, inception]

# this cell must sit above loader, as image resizing inside transform depends on `input_size`.

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ 
        Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ 
        Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ 
        VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ 
        Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ 
        Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Loader
class NihDataset(Dataset):
    def __init__(self, dataframe, root_dir, label, transform=None):
        """
        label: column name of the label of interest, e.g. 'Pleural Effusion'.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.label = label
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.loc[idx, 'Image Index'])
        # via .getband(), some images are know to have 4 channels.
        # Here we may convert them to 1-channel (grayscale) or 3-channel (RGB) depending on the model.
        image = Image.open(img_name).convert('RGB')
        target = self.dataframe.loc[idx, self.label]
            
        if self.transform:
            image = self.transform(image)
        
        return image, target

# Data loaders to return batch of images
def load_data(dataframe, root_dir, label, transform=None, batch_size=32, shuffle=True, num_workers=0):
    '''
    Data Loader with batch loading and transform.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_data = NihDataset(dataframe, root_dir, label, transform=transform)
    pin = device=='cpu'
    loader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)
    return loader

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, verbose=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_time = time.time()
    best_val_roc = 0.0
    train_losses, val_losses, val_rocs = [], [], []
    for epoch in range(num_epochs):
        print("="*40)
        model.train()
        print(f"Epoch {epoch+1}")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics every 10 batches
            running_loss += loss.item()
            if verbose:
                if i % 10 == 9:    # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))
        train_losses.append(running_loss/len(train_loader))  # keep trace of train loss in each epoch
        print(f'Time elapsed: {(time.time()-start_time)/60.0:.1f} minutes.')
        
        # validate every epoch
        print("Validating...")
        val_loss, val_auc, _, _, _ = eval_model(model, valid_loader, criterion)
        val_rocs.append(val_auc)
        
        # save all models
        MODEL_PATH = f'../models/{int(start_time)}_epoch_{epoch}.pth'
        torch.save(model.state_dict(), MODEL_PATH)
        best_model_path = MODEL_PATH
        
        # save the best model
        if val_auc > best_val_roc:
            best_val_roc = val_auc
            MODEL_PATH = f'../models/{int(start_time)}_bestroc.pth'
            torch.save(model.state_dict(), MODEL_PATH)
            print("Model saved!")
            
        val_losses.append(val_loss/len(valid_loader))  # keep trace of validation loss in each epoch
        print("Epoch: {}/{}.. ".format(epoch+1, num_epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Validation Loss: {:.3f}.. ".format(val_losses[-1]))
    return model, train_losses, val_losses, best_val_roc, val_rocs, best_model_path

def eval_model(model, loader, criterion):
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # validate every epoch
    loss = 0.0
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        # empty tensors to hold results
        Y_prob, Y_true, Y_pred = [], [], []
        for inputs, labels in loader:
            probs = model(inputs.to(device)) # prediction probability
            labels = labels.type(torch.LongTensor).to(device)  # true labels
            _, predicted = torch.max(probs, dim=1)
            # stack results
            Y_prob.append(probs[:,-1])  # probability of positive class
            Y_true.append(labels)
            Y_pred.append(predicted)
            
            loss += float(criterion(probs, labels))

    # convert to numpy
    Y_prob = torch.cat(Y_prob).detach().cpu().numpy()
    Y_pred = torch.cat(Y_pred).detach().cpu().numpy()
    Y_true = torch.cat(Y_true).detach().cpu().numpy()

    # TODO: use desired metrics here
    print(f"ROC: {roc_auc_score(Y_true, Y_prob):.3f}")
    fpr, tpr, _ = roc_curve(Y_true, Y_prob)
    auc = roc_auc_score(Y_true, Y_prob)
    #plt.figure()
    #plt.plot(fpr,tpr,label=f"Validation, ROCAUC={auc:3f}")
    #plt.legend(loc=4)
    #plt.show()
    
    print(classification_report(Y_true, Y_pred))
    #plt.figure()
    #cm = confusion_matrix(Y_true, Y_pred)  # confusion matrix
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot()
         
    return loss, auc, Y_prob, Y_pred, Y_true
