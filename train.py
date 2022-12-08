import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import utils
import torch
import torchvision
import numpy as np
from pytorch_msssim import msssim
from tqdm import tqdm
from torch.optim import Adam
from net import DenseFuse_net
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from option import opt

def train():
    ## prepare data
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = utils.MyPolDataset(opt.traindata_dir, transform=data_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.bs, shuffle=False)

    ## net
    densefuse_model = DenseFuse_net(opt.input_channel, opt.output_channel)
    densefuse_model.initialize_weights()
    densefuse_model = densefuse_model.to(device=opt.device)

    ## Optimizer
    optimizer = Adam(densefuse_model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=opt.lr_decay_rate)

    ## log
    writer = SummaryWriter('logs/')

    print('Start training.....')
    for epoch in range(opt.epochs):
        loss = 0.
        batch_train = 0.
        densefuse_model.train()
        with tqdm(train_loader, unit="batch") as tepoch:        ##set tqdm
            for images_s0, images_dolp in tepoch:
                optimizer.zero_grad()
                mode = np.random.permutation(8)
                images_s0 = images_s0.to(device=opt.device)
                images_dolp = images_dolp.to(device=opt.device)
                images_s0 = utils.data_augmentation(images_s0,mode[0])
                images_dolp = utils.data_augmentation(images_dolp,mode[0])              
                outputs = densefuse_model(images_s0, images_dolp)

                ## ssim_loss
                ssim_loss_value = msssim(images_s0, images_dolp, outputs, normalize=True)
                ## intensity_loss
                w1 = torch.exp(images_s0/0.1)/(torch.exp(images_s0/0.1) + torch.exp(images_dolp/0.1))
                intensity_loss = torch.mean(w1 * ((images_s0-outputs)**2)) + torch.mean((1 - w1) * ((images_dolp-outputs)**2))
                ## total loss
                total_loss = 1*ssim_loss_value +  10*intensity_loss
                total_loss.backward()
                optimizer.step
                
                loss = loss + total_loss
                batch_train = batch_train + 1
                tepoch.set_description_str('Epoch: {:d},loss: {:f},total_loss: {:f}'.format(epoch+1, total_loss, loss/batch_train))  

        scheduler.step()      
        writer.add_scalar('train_loss', loss/batch_train, global_step=epoch+1)
        
        ## save model
        densefuse_model.eval()
        save_model_path = './models/model_'+str(epoch+1)+'.pth'
        torch.save(densefuse_model.state_dict(), save_model_path)
        
    writer.close()
    print("\nDone, trained model saved at", save_model_path)

if __name__ == "__main__":
    train()