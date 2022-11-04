import torch 
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

from args import parser 
from resnet import ResNet

import torch.optim as optim
import torch.nn as nn  

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(epoch, model, optimizer, filename):
    state ={
        'Epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()

        # epcoh
        # loss 
    
    }
    
    torch.save(state, filename)

def get_dir_name():
    split_symbol = '~' if os.name == 'nt' else ':'
    model_name_template = split_symbol.join(['S:{}_batch_size', '{}_layer', '{}_id'])
    model_name = model_name_template.format(arg.batch_size, arg.layer, arg.id)
    
    dir_name = os.path.join(model_name)
            
    # print(dir_name)

    return dir_name 


def meanSubtract(dataset):
    mean = [np.mean(x.numpy(), axis=(1,2)) for x, _ in dataset]
    std = [np.std(x.numpy(), axis=(1,2)) for x, _ in dataset]

    meanR = np.mean([m[0] for m in mean])
    meanG = np.mean([m[1] for m in mean])
    meanB = np.mean([m[2] for m in mean])
    
    stdR = np.mean([s[0] for s in std])
    stdG = np.mean([s[1] for s in std])
    stdB = np.mean([s[2] for s in std])

    return meanR, meanG, meanB, stdR, stdG, stdB

def get_data(mode):
    # load tranin dataset : CIFAR-10
    batch_size = arg.batch_size
   
    if(mode=='train'):
        meanR, meanG, meanB = 0.49139965, 0.48215845, 0.4465309
        stdR, stdG, stdB = 0.20220213, 0.19931543, 0.20086348    
        
    elif (mode =='test'):
        meanR, meanG, meanB = 0.4942142, 0.48513138, 0.45040908
        stdR, stdG, stdB = 0.20189482, 0.19902097, 0.20103233    
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((meanR, meanG, meanB),
                                                         (stdR, stdG, stdB))])   
    
    # load train datasaet : CIFAR-10
    if(mode=='train'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                download=True,
                                                transform=transform)
        
        train, valid = torch.utils.data.random_split(trainset, [45000, 5000])
        dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                                shuffle=True,
                                                )   
        # validation set 
        validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, 
                                                shuffle=True,
                                                ) # check batch size 
    # load test datasaet : CIFAR-10
    elif (mode =='test'):
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True,
                                            transform=transform)     
        dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False) 

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    print("Done with", mode,  "data loading")
    
    if(mode=='train'):
        return dataloader, validloader, classes, 
    elif (mode =='test'):
        return dataloader, classes
    
def train(trainloader, validloader, classes):
    
    model = ResNet(arg.layer).to(device)
    # summary(model, (3, 32, 32))
    # print(summary)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= arg.lr , weight_decay= 0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2000)    # 20 epoch
    
    epoch_num = arg.epoch
    # total training 45000, batch 256, mini batch 45000/256 => 176
    mini_batch = int(45000 / arg.batch_size) # training data 45000
    # total validation 5000, batch 256, mini batch 5000/256 => 20
    mini_batch_val = int(5000 / arg.batch_size) # training data 45000
    
    if(arg.mode == 'resume'):
        model_path = os.path.join('./model/', PATH)
        file_path = model_path + '/ResNet.pth'
        
        print("check point ", file_path)
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        tb_pth_train = os.path.join('./logs/train/', PATH)
        tb_pth_valid = os.path.join('./logs/valid/', PATH)
        
    else:
        # mkdir modelpath 
        model_path = os.path.join('./model/', PATH)
        print("model path", model_path)
        
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        
        file_path = model_path + '/ResNet.pth'
        print("FINAL: ", file_path)
        
        # train
        tb_pth_train = os.path.join('./logs/train/', PATH)
        if not os.path.isdir(tb_pth_train):
            os.makedirs(tb_pth_train)
        
        # print(tb_pth_train)
        
        # valid 
        tb_pth_valid = os.path.join('./logs/valid/', PATH)
        if not os.path.isdir(tb_pth_valid):
            os.makedirs(tb_pth_valid)        

    for epoch in range(epoch_num):
        # train 
        writer = SummaryWriter(tb_pth_train)      # tensorboard 
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):   # starts from index 0 
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()

            # accuracy
            _, predicted = torch.max(output.data, 1) # top-1 : check paper
            total+= labels.size(0) 
            correct += (predicted == labels).sum().item()
            
            if i % (mini_batch + 1) == mini_batch:    # print every 176 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}], train loss: {running_loss / (mini_batch + 1):.3f},  train accuracy: {(100 * correct // total):.2f}%,' , end=' ')
                writer.add_scalar("Loss/train", running_loss / (mini_batch + 1), epoch + 1)
                writer.add_scalar("Accuracy/train", 100 * correct // total, epoch + 1)
                running_loss = 0.0
                
        writer.close()
        
        
        # validation 
        writer = SummaryWriter(tb_pth_valid)      # tensorboard 
        validation_loss = 0.0
        
        correct = 0
        total = 0
       
        model.eval()                             # check placement 
        criterion.eval()                         # check placement 
        with torch.no_grad():   
            for j, val_data in enumerate(validloader, 0):
                val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_out = model(val_inputs)
                
                val_loss = criterion(val_out, val_labels)
                validation_loss += val_loss.item()
                
                # validation accuracy 
                _, predicted = torch.max(val_out.data, 1) # top-1 : check paper
                total+= val_labels.size(0) 
                correct += (predicted == val_labels).sum().item()               
                
                if j % (mini_batch_val + 1) == mini_batch_val:    # print every 20 mini-batches
                    print(f' val loss: {validation_loss / (mini_batch_val + 1):.3f}, valid accuracy: {(100 * correct // total):.2f}%, lr: {optimizer.param_groups[0]["lr"]:.6f}')
                    writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch + 1)
                    writer.add_scalar("Loss/val", validation_loss / (mini_batch_val + 1), epoch + 1)
                    writer.add_scalar("Accuracy/val", 100 * correct // total, epoch + 1)
                    validation_loss = 0.0
                    
                scheduler.step(validation_loss) 
            writer.close()
        
        model.train()                           # check placement 
        criterion.train()                       # check placement 
        
        save_checkpoint(epoch, model, optimizer, file_path) # save per epoch
        
    print('Finished Training')
    writer.close()
    
    return file_path
    
def test(testloader, save_model):
    print("Start the test!")
    
    # tensorboard 
    # writer = SummaryWriter('./logs/test')

    test_model = ResNet(arg.layer).to(device)
    optimizer = optim.SGD(test_model.parameters(), lr= arg.lr , weight_decay= 0.0001, momentum=0.9)
    
    checkpoint = torch.load(save_model)
    test_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # test_model.load_state_dict(torch.load(save_model))
    
    # test 
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            
            images, labels = data[0].to(device), data[1].to(device)     # error : declare CUDA
            
            outputs= test_model(images)
            _, predicted = torch.max(outputs.data, 1)                   # top-1 : check paper
            total+= labels.size(0) 
            correct += (predicted == labels).sum().item()
            
    print(f'Accuracy of the network : {(100 * correct // total):.2f}%')
    print(f'Error of the network : {(100 - (100 * correct // total)):.2f} %')
    
    # writer.close()

def run(): 
    # parse the arguments
    global arg, device, model, PATH 
    
    arg = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    # load the train a& test data
    train_load, valid_load, classes = get_data('train')
    test_load, classes = get_data('test')
    
    PATH = get_dir_name()
    
    save_model= train(train_load, valid_load, classes)
    test(test_load, save_model)


# save epoch and loss
# data augmentation
# paper spec 
# github
# save best accuracy model 

def main():
    run()

main()
