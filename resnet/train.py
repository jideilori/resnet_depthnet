import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils 
from utils import AverageMeter,SummaryWriter,LogProgress
from load_data import getTrainingTestingData
from loss import ssim
from model import Resnet_UNet
from utils import DepthNorm


epochs=1 
learning_rate = 0.0001
batch_size =4

# Create model
model = Resnet_UNet().cuda()
print('Model created.')

# Training parameters
optimizer = torch.optim.Adam( model.parameters(), learning_rate )
batch_size = batch_size
prefix = 'resnet_' + str(batch_size)

# Load data
train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

# Logging
writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, learning_rate, epochs, batch_size), flush_secs=30)

# Loss
l1_criterion = nn.L1Loss()


# Start training...
for epoch in range(epochs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    N = len(train_loader)

    # Switch to train mode
    model.train()

    end = time.time()

    for i, sample_batched in enumerate(train_loader):
        optimizer.zero_grad()

        # Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        # Normalize depth
        depth_n = DepthNorm( depth )

        # Predict
        output = model(image)

        # Compute the loss
        l_depth = l1_criterion(output, depth_n)
        l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

        loss = (1.0 * l_ssim) + (0.1 * l_depth)

        # Update step
        losses.update(loss.data.item(), image.size(0))
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
    
        # Log progress
        niter = epoch*N+i
        if i % 5 == 0:
            # Print to console
            print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
            'ETA {eta}\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})'
            .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

            # Log to tensorboard
            writer.add_scalar('Train/Loss', losses.val, niter)

        if i % 300 == 0:
            LogProgress(model, writer, test_loader, niter)

    # Record epoch's intermediate results
    LogProgress(model, writer, test_loader, niter)
    writer.add_scalar('Train/Loss.avg', losses.avg, epoch)