import cv2
import torch
import numpy as np
# You should import your dataset tools
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
from net.T_Net import TModel
import os
from simulation.diff_grad import grad

parse = argparse.ArgumentParser('')
parse.add_argument('--batch_size' , type=int , default=4 , help='')
parse.add_argument('--dataset' , type=str , default='Chikusei' , help='')
parse.add_argument('--dataroot' , type=str , default='../DATA/Chikusei/' , help='')
parse.add_argument('--save_folder', default='./model/Chikusei/T_Net/', help='Directory to keep training outputs.')
parse.add_argument('--epochs', type=int , default=401, help='')
parse.add_argument('--lr', type=float , default=1e-4, help='')
parse.add_argument('--hsi_dim', type=int , default=128, help='')
parse.add_argument('--msi_dim', type=int , default=4, help='')
opt = parse.parse_args()

lr = opt.lr
model = TModel(opt.hsi_dim , opt.msi_dim).cuda()
mse_loss = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 400 , eta_min= 1e-6)

def save_checkpoints(model , epoch):
     model_dir = opt.save_folder + 'T_Net_{}.pth'.format(epoch)
     if not os.path.exists(opt.save_folder):
         os.makedirs(opt.save_folder)

     checkpoint = {
              'net' : model.state_dict(),
              'optimizer': optimizer.state_dict(),
               "epoch": epoch,
               "lr":lr
     }
     torch.save(checkpoint , model_dir)
     print("Checkpoint saved to {}".format(model_dir))


step = 50



def train(Resume = False):

    start_epoch = 1

    if Resume == True:
        checkpoint = torch.load(opt.save_folder + '\T_Net_{}.pth'.format(1000))
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('===> model is already !')


    ### dataset load 
    train_datasets = ''
    val_datasets = ''
    train_loader = DataLoader(train_datasets , batch_size=opt.batch_size , shuffle=True)
    val_loader = DataLoader(val_datasets , batch_size=opt.batch_size)

    print('data_length: ',len(train_datasets))


    model.train()
    for epoch in range(start_epoch , opt.epochs):

        for iteration , data in enumerate(train_loader , 1):
            _ , MSI , HSI = data[0].cuda() , data[1].cuda() , data[2].cuda()
            global_grad , local_x_grad , local_y_grad = model(HSI , MSI)

            ## global_loss: decomposed into the losses in the x and y directions
            global_loss = mse_loss(global_grad[0] , global_grad[1]) + mse_loss(global_grad[2] , global_grad[3])

            ## local_x_loss: four sub-view losses along the x-direction
            local_x_loss = mse_loss(local_x_grad[0] , local_x_grad[1]) + mse_loss(local_x_grad[2] , local_x_grad[3]) + mse_loss(local_x_grad[4] , local_x_grad[5]) + \
                           mse_loss(local_x_grad[6] , local_x_grad[7])

            ## local_y_loss: four sub-view losses along the y-direction
            local_y_loss = mse_loss(local_y_grad[0] , local_y_grad[1]) + mse_loss(local_y_grad[2] , local_y_grad[3]) + mse_loss(local_y_grad[4] , local_y_grad[5]) + \
                           mse_loss(local_y_grad[6] , local_y_grad[7])

            total_loss = global_loss + local_x_loss + local_y_loss

            optimizer.zero_grad()

            total_loss.backward()
            optimizer.step()

            if iteration % 4 == 0:
            # if iteration % 1 == 0:
                # log_value('Loss', loss.data[0], iteration + (epoch - 1) * len(training_data_loader))
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(train_loader),total_loss.item()))

        
        scheduler.step()

        if epoch % 50 == 0:
            save_checkpoints(model , epoch)

        ### Val
            model.eval()
            with torch.no_grad():
                for iteration, data in enumerate(val_loader, 1):
                    _ , MSI , HSI = data[0].cuda() , data[1].cuda() , data[2].cuda()
                    global_grad, local_x_grad, local_y_grad = model(HSI, MSI)

                    results = global_grad + local_x_grad + local_y_grad

                    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(15, 6))
                    for ax, res, idx in zip(axes.flat, results, range(1, 21)):
                        res = res.detach().cpu()
                        if res.dtype is torch.uint8:
                            res = res.float() / 255.0
                        else:
                            res = res.float()
                            res -= res.min()
                            res /= (res.max() + 1e-8)
                            res =  res.clamp(0, 1).numpy()
                        res = res[0].transpose(1 , 2 , 0)
                        res = np.mean(res , axis=2)

                        ax.imshow(res , cmap='gray')
                        ax.axis('off')
                        ax.set_title(f"Image {idx}", fontsize=10)
                    plt.tight_layout()
                    plt.show()

def test():

    test_datasets = ''
    test_loader = DataLoader(test_datasets, batch_size=opt.batch_size)

    checkpoint = torch.load(opt.save_folder + '\T_Net_{}.pth'.format(400))
    model.load_state_dict(checkpoint['net'])
    print('===> model is already !')

    model.eval()
    with torch.no_grad():
        for iteration, data in enumerate(test_loader, 1):
            _, MSI, HSI = data[0].cuda(), data[1].cuda(), data[2].cuda()
            global_grad, local_x_grad, local_y_grad = model(HSI, MSI)

            results = global_grad + local_x_grad + local_y_grad

            fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(15, 6))
            for ax, res, idx in zip(axes.flat, results, range(1, 21)):
                res = res.detach().cpu()
                if res.dtype is torch.uint8:
                    res = res.float() / 255.0
                else:
                    res = res.float()
                    res -= res.min()
                    res /= (res.max() + 1e-8)
                    res = res.clamp(0, 1).numpy()
                res = res[0].transpose(1, 2, 0)
                res = np.mean(res, axis=2)
                
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':

    # train()
    test()














