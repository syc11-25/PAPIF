import torch,os,sys,torchvision,argparse
import torch,warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='Automatic detection')
parser.add_argument('--bs', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='lr decay rate')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--input_channel', type=int, default=1)
parser.add_argument('--output_channel', type=int, default=1)
parser.add_argument('--traindata_dir', type=str, default='./dataset/train')
parser.add_argument('--testdata_dir', type=str, default='./dataset/test')
parser.add_argument('--testmodel_dir', type=str, default="./models/model_99.pth")
parser.add_argument('--resume', type=bool,default=False)


opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'

print(opt)

if not os.path.exists('models'):
	os.mkdir('models')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('output'):
	os.mkdir('output')

