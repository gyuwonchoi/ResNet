import torch 
import torchvision
import torchvision.transforms as transforms
import numpy as np

from args import parser 
from resnet import ResNet

import torch.optim as optim
import torch.nn as nn  

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

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
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True,
                                                )   
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
    return dataloader, classes
    
def train(trainloader, classes):
    
    # tensorboard 
    writer = SummaryWriter('./logs/train')
    
    model = ResNet(arg.layer).to(device)
    summary(model, (3, 32, 32))
    print(summary)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= arg.lr , weight_decay= 0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50)    # lr scheduler : adjust param

    epoch_num = arg.epoch
    mini_batch = int(50000 / arg.batch_size)
    print(mini_batch)
    
    for epoch in range(epoch_num):
        running_loss = 0.0
        
        # total training 50000, batch 256, 50000/256 = 196
        for i, data in enumerate(trainloader, 0): # starts from index 0 
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()

            if i % (mini_batch + 1) == mini_batch:    # print every 256 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}], loss: {running_loss / (mini_batch + 1):.3f}, lr: {optimizer.param_groups[0]["lr"]:.6f}')
                writer.add_scalar("Loss/train", running_loss / (mini_batch + 1), epoch + 1)
                running_loss = 0.0
                
        scheduler.step(running_loss) # put validation loss
        
    print('Finished Training')
    
    # model save 
    # model_name_template = ''.join(['{}_batch_size', '{}_layer', '{}_epoch', '{}_dataset'])
    # model_name = model_name_template.format(arg.batch_size, arg.layer, arg.epoch, arg.dataset)
    PATH = './model/cifar10.pth'
    torch.save(model.state_dict(), PATH)
    print('Finished model saving')
    
    writer.close()
    
def test(testloader, classes):
    print("Start the test!")
    
    # tensorboard 
    # writer = SummaryWriter('./logs/test')
    
    # model declariation
    test_model = ResNet(arg.layer).to(device)
    
    # should delete the path 
    # model_name_template = './model/'.join(['{}_batch_size', '{}_layer', '{}_epoch', '{}_dataset'])
    # model_name = model_name_template.format(arg.batch_size, arg.layer, arg.epoch, arg.dataset)
    PATH = './model/cifar10.pth'
    test_model.load_state_dict(torch.load(PATH))
    
    # test 
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            
            # error : declare CUDA
            images, labels = data[0].to(device), data[1].to(device)
            
            outputs= test_model(images)
            _, predicted = torch.max(outputs.data, 1) # top-1 : check paper
            total+= labels.size(0) 
            correct += (predicted == labels).sum().item()
            
    print(f'Accuracy of the network : {100 * correct // total} %')
    print(f'Error of the network : {100 - (100 * correct // total)} %')
    
    # writer.close()


def run(): 
    # parse the arguments
    global arg, device, model
    
    arg = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    # load the train a& test data
    train_load, classes = get_data('train')
    test_load, classes = get_data('test')
    
    train(train_load, classes)
    test(test_load, classes)


# 45k/5k train/val split

# tensorboardx : accuracy, error, loss 
# save model proper
# print training  accuracy, loss, lr 

# data augmentation : padding, and crop

# percent .xx format 

# add learning scheduler 
# paper spec 

# validation set??

def main():
    run()

main()
