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
    model_name_template = split_symbol.join(['S:{}_mini_batch', '{}_layer', '{}_id'])
    model_name = model_name_template.format(arg.mini_batch, arg.layer, arg.id)
    
    dir_name = os.path.join(model_name)
    
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
    # batch_size_train = 45000 / arg.batch_num  # 351.5 
    # batch_size_valid = 5000  / arg.batch_num  # 39.06
    # batch_size_test  = 10000 / arg.batch_num  # 78.125
    
    batch_size_train = arg.mini_batch  # 128 
    batch_size_valid = arg.mini_batch  # 128
    batch_size_test  = arg.mini_batch  # 128
    
    if(mode=='train'):
        meanR, meanG, meanB = 0.49139965, 0.48215845, 0.4465309
        stdR, stdG, stdB = 0.20220213, 0.19931543, 0.20086348    
        
    elif (mode =='test'):
        meanR, meanG, meanB = 0.4942142, 0.48513138, 0.45040908
        stdR, stdG, stdB = 0.20189482, 0.19902097, 0.20103233    
    
    transform_train = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((meanR, meanG, meanB),
                                                         (stdR, stdG, stdB)),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip()])   
    
    transform_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((meanR, meanG, meanB),
                                                         (stdR, stdG, stdB))])   
    
    # load train datasaet : CIFAR-10
    if(mode=='train'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                download=True,
                                                transform=transform_train)
        
        train, valid = torch.utils.data.random_split(trainset, [45000, 5000])
        dataloader = torch.utils.data.DataLoader(train, batch_size = int(batch_size_train),
                                                shuffle=True,
                                                )   
        # validation set 
        validloader = torch.utils.data.DataLoader(valid, batch_size = int(batch_size_valid), 
                                                shuffle=True,
                                                ) # check batch size 
    # load test datasaet : CIFAR-10
    elif (mode =='test'):
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True,
                                            transform=transform_test)     
        dataloader = torch.utils.data.DataLoader(testset, batch_size = int(batch_size_test),
                                                shuffle=False) 

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if(mode=='train'):
        return dataloader, validloader, classes
    
    elif (mode =='test'):
        return dataloader, classes
    
def test_in_train(testloader, save_model, epoch, tb_pth_test):
    writer = SummaryWriter(tb_pth_test)
    # test_model = ResNet(arg.layer).to(device)
       
    # test 
    correct = 0
    total = 0
    
    # test_model.eval()      # enable for consistant performance 
    with torch.no_grad():
        for data in testloader:
            
            images, labels = data[0].to(device), data[1].to(device)     # error : declare CUDA
            
            outputs= save_model(images)
            _, predicted = torch.max(outputs.data, 1)                   # top-1 : check paper
            total+= labels.size(0) 
            correct += (predicted == labels).sum().item()
            
    writer.add_scalar("Error/test", 100.0 - 100.0 * correct / total, epoch + 1)
    # print(f'Accuracy: {(100.0 * correct / total):.2f}%, Error : {(100.0 - (100.0 * correct / total)):.2f}% ')
    if (epoch % 50 ==0):
        print('Accuracy: ', 100.0 * correct / total, '%, Error : ', 100.0 - (100.0 * correct / total), '%')
    writer.close()
    
def validate(model_, validloader, tb_pth_valid, epoch, batch_num_val):
    writer = SummaryWriter(tb_pth_valid)           # tensorboard 
    
    model = ResNet(arg.layer).to(device)
    model = model_
    optimizer = optim.SGD(model.parameters(), lr= arg.lr , weight_decay= 0.0001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    validation_loss = 0.0    
    correct = 0.0
    total = 0.0
       
    with torch.no_grad():   
        # model.eval()                             
        for j, val_data in enumerate(validloader, 0):
            val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_out = model(val_inputs)
            
            val_loss = criterion(val_out, val_labels)
            validation_loss += val_loss.item()
            
            # validation accuracy 
            _, predicted = torch.max(val_out.data, 1) 
            total+= val_labels.size(0) 
            correct += (predicted == val_labels).sum().item()               
            
            if j % (batch_num_val + 1) == batch_num_val:    
                print(f' val loss: {validation_loss / (batch_num_val + 1):.3f}, valid accuracy: {(100.0 * correct / total):.2f}%, lr: {optimizer.param_groups[0]["lr"]:.6f}')
                writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch + 1)
                writer.add_scalar("Loss/val", validation_loss / (batch_num_val + 1.0), epoch + 1)
                writer.add_scalar("Accuracy/val", 100.0 * correct / total, epoch + 1)
                writer.add_scalar("Error/val", 100.0 - 100.0 * correct / total, epoch + 1)
                validation_loss = 0.0
                
        writer.close()

def train(trainloader, validloader, classes, test_load):
    
    model = ResNet(arg.layer).to(device)
    summary(model, (3, 32, 32))
    print(summary)
    
    lr_ = arg.lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= lr_, weight_decay= 0.0001, momentum=0.9)
    
    epoch_num = arg.epoch
    batch_num = int (45000 / arg.mini_batch) 
    batch_num_val = int (5000 / arg.mini_batch)
    
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
        
        tb_pth_test = os.path.join('./logs/test/', PATH)
        if not os.path.isdir(tb_pth_test):
            os.makedirs(tb_pth_test)        
        
    # ================ train ================
    # 
    # iteration= epoch x mini-batch 
    #  
    # =======================================
    for epoch in range(epoch_num):
        writer = SummaryWriter(tb_pth_train)      # tensorboard 
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        
        if(epoch == 100): # 250 175
            lr_ = 0.01
        elif(epoch == 150): # 375 262
            lr_ = 0.001
        
        # add to test lr reduction
        optimizer = optim.SGD(model.parameters(), lr= lr_, weight_decay= 0.0001, momentum=0.9)
        
        model.train()                               
        for i, data in enumerate(trainloader, 0):   # 45000 / 128
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()

            # accuracy
            _, predicted = torch.max(output.data, 1) 
            total+= labels.size(0) 
            correct += (predicted == labels).sum().item()
            
            if i % (batch_num + 1) == batch_num:    
                print(f'[{epoch + 1} / {epoch_num}], train loss: {running_loss / (batch_num + 1.0):.3f},  train accuracy: {(100.0 * correct / total):.2f}%,' , end=' ')       
                writer.add_scalar("Loss/train", running_loss / (batch_num + 1.0), epoch + 1)
                writer.add_scalar("Accuracy/train", 100.0 * correct / total, epoch + 1)
                running_loss = 0.0

        # scheduler.step(running_loss)                 
        writer.close()

        validate(model, validloader, tb_pth_valid, epoch, batch_num_val)
    # ================ validation ================
        # writer = SummaryWriter(tb_pth_valid)           # tensorboard 
        # validation_loss = 0.0
        
        # correct = 0.0
        # total = 0.0
       
        # with torch.no_grad():   
        #     # model.eval()                             
        #     for j, val_data in enumerate(validloader, 0):
        #         val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
        #         val_out = model(val_inputs)
                
        #         val_loss = criterion(val_out, val_labels)
        #         validation_loss += val_loss.item()
                
        #         # validation accuracy 
        #         _, predicted = torch.max(val_out.data, 1) 
        #         total+= val_labels.size(0) 
        #         correct += (predicted == val_labels).sum().item()               
                
        #         if j % (batch_num_val + 1) == batch_num_val:    
        #             print(f' val loss: {validation_loss / (batch_num_val + 1):.3f}, valid accuracy: {(100.0 * correct / total):.2f}%, lr: {optimizer.param_groups[0]["lr"]:.6f}')
        #             writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch + 1)
        #             writer.add_scalar("Loss/val", validation_loss / (batch_num_val + 1.0), epoch + 1)
        #             writer.add_scalar("Accuracy/val", 100.0 * correct / total, epoch + 1)
        #             writer.add_scalar("Error/val", 100.0 - 100.0 * correct / total, epoch + 1)
        #             validation_loss = 0.0
                    
        #     writer.close()
        
        save_checkpoint(epoch, model, optimizer, file_path) # save per epoch
        test_in_train(test_load, model, epoch, tb_pth_test)
            
        
    print('Finished Training')
    writer.close()
    
    return file_path
    
def test(testloader, save_model):
    print("Start the test!")
    
    test_model = ResNet(arg.layer).to(device)
    optimizer = optim.SGD(test_model.parameters(), lr= arg.lr , weight_decay= 0.0001, momentum=0.9)
    
    checkpoint = torch.load(save_model)
    test_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

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
            
    print('Accuracy of the network :', 100.0 * correct / total, '%')
    print('Error of the network :', 100.0 - 100.0 * correct / total, '%')


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
    
    save_model= train(train_load, valid_load, classes, test_load)
    test(test_load, save_model)

    # test(test_load, './model/S:128_mini_batch:20_layer:7_id/ResNet.pth')
    
def main():
    run()

main()
