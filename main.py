from __future__ import print_function
import argparse
from math import log10

import os

import braceexpand
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from dbpn_v1 import Net as DBPNLL
from dbpns import Net as DBPNS
from dbpn_iterative import Net as DBPNITER
from data import get_training_set
import pdb
import socket
import time
import webdataset as wds
from functools import partial

from fmow_superres import fmow_preprocess_train

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./Dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', action='store_true', default=False)
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--pretrained_sr', default='MIX2K_LR_aug_x4dl10DBPNITERtpami_epoch_399.pth',
                    help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='tpami_residual_filter8', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)

        if opt.residual:
            prediction = prediction + bicubic

        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        if (iteration % opt.snapshots) == 0:
            checkpoint(iteration)

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(training_data_loader), loss.data,
                                                                                 (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_path = os.path.join(opt.save_folder, f"itr_{epoch}.pth")
    # model_out_path = opt.save_folder + opt.train_dataset + hostname + opt.model_type + opt.prefix + "_epoch_{}.pth".format(
    #     epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

shards = (
        list(braceexpand.braceexpand(
            '/atlas2/data/satlas/fmow-512-sentinel-paired/fmow-512-sentinel-paired-train-000000-019999.tar')) +
        list(braceexpand.braceexpand(
            '/atlas2/data/satlas/fmow-512-sentinel-paired/fmow-512-sentinel-paired-train-020000-039999.tar')) +
        list(braceexpand.braceexpand(
            '/atlas2/data/satlas/fmow-512-sentinel-paired/fmow-512-sentinel-paired-train-040000-059999.tar')) +
        list(braceexpand.braceexpand(
            '/atlas2/data/satlas/fmow-512-sentinel-paired/fmow-512-sentinel-paired-train-060000-079999.tar')) +
        list(braceexpand.braceexpand(
            '/atlas2/data/satlas/fmow-512-sentinel-paired/fmow-512-sentinel-paired-train-080000-099999.tar')) +
        list(braceexpand.braceexpand(
            '/atlas2/data/satlas/fmow-512-sentinel-paired/fmow-512-sentinel-paired-train-100000-119999.tar')) +
        list(braceexpand.braceexpand(
            '/atlas2/data/satlas/fmow-512-sentinel-paired/fmow-512-sentinel-paired-train-120000-139999.tar')) +
        list(braceexpand.braceexpand(
            '/atlas2/data/satlas/fmow-512-sentinel-paired/fmow-512-sentinel-paired-train-140000-159999.tar'))
)
train_set = wds.DataPipeline(
    wds.ResampledShards(shards),
    wds.tarfile_to_samples(),
    wds.shuffle(100, initial=100),
    wds.decode(),
    partial(fmow_preprocess_train, patch_size=opt.patch_size, lowres=64, highres=512, is_train=True),
).with_length(10000)
# train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, )

print('===> Building model ', opt.model_type)
if opt.model_type == 'DBPNLL':
    model = DBPNLL(in_channels=13, out_channels=3, base_filter=64, feat=256, num_stages=10,
                   scale_factor=opt.upscale_factor)
elif opt.model_type == 'DBPN-RES-MR64-3':
    model = DBPNITER(in_channels=13, out_channels=3, base_filter=64, feat=256, num_stages=3,
                     scale_factor=opt.upscale_factor)
else:
    model = DBPN(in_channels=13, out_channels=3, base_filter=64, feat=256, num_stages=7,
                 scale_factor=opt.upscale_factor)

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        # model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch)

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch + 1) % (opt.nEpochs / 2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch + 1) % (opt.snapshots) == 0:
        checkpoint(epoch)