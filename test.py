import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import utils
import torch
import torchvision
from net import DenseFuse_net
from torch.utils.data import DataLoader
from imageio import imsave
import time
from option import opt

def test():
    ##prepare data
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = utils.MyPolDataset(opt.testdata_dir, transform=data_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

    ##load net
    with torch.no_grad():
        print('testing')
        test_model = DenseFuse_net(opt.input_channel, opt.output_channel)
        test_model.load_state_dict(torch.load(opt.testmodel_dir))
        test_model = test_model.to(device=opt.device)
        test_model.eval()
        begin_time = time.time()
        for i, (images_s0,images_dolp) in enumerate(train_loader):
            images_s0 = images_s0.to(device=opt.device)
            images_dolp = images_dolp.to(device=opt.device)
            img_fusion = test_model(images_s0,images_dolp)   
            torchvision.utils.save_image(img_fusion,'./output/'+str(i)+'.png')        
    proc_time = time.time() - begin_time
    print('Total processing time: {:.3}s'.format(proc_time))

if __name__ == "__main__":
    test()