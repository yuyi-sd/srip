import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search_miniimagenet import Network
from architect import Architect

from miniImagenet import miniImagenetOneShotDataset

parser = argparse.ArgumentParser("miniImagenet")
parser.add_argument('--set', type=str, default='miniImagenet', help='dataset name')
parser.add_argument('--batch_size', type=int, default=24, help='batch size')
parser.add_argument('--classes_per_set', type=int, default=5, help='way')
parser.add_argument('--samples_per_class', type=int, default=1, help='shot')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

batch_size = args.batch_size
fce = True
classes_per_set = args.classes_per_set
samples_per_class = args.samples_per_class

total_train_batches = 500
total_val_batches = 500
total_test_batches = 200
best_val_acc = 0.0
dataroot = 'miniimagenet'
dataTrain = miniImagenetOneShotDataset(dataroot = dataroot, 
                                        type = 'train', 
                                        nEpisodes = total_train_batches * batch_size * args.epochs, 
                                        classes_per_set = classes_per_set, 
                                        samples_per_class = samples_per_class)
dataVal = miniImagenetOneShotDataset(dataroot = dataroot, 
                                        type = 'val', 
                                        nEpisodes = total_train_batches * batch_size * args.epochs, 
                                        classes_per_set = classes_per_set, 
                                        samples_per_class = samples_per_class)
dataTest = miniImagenetOneShotDataset(dataroot = dataroot, 
                                        type = 'test', 
                                        nEpisodes = total_test_batches * batch_size, 
                                        classes_per_set = classes_per_set, 
                                        samples_per_class = samples_per_class)



log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, classes_per_set, args.layers, criterion, args.batch_size)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  train_loader = torch.utils.data.DataLoader(dataTrain, batch_size = batch_size, shuffle = True, num_workers = 4)
  val_loader = torch.utils.data.DataLoader(dataVal, batch_size = batch_size, shuffle = True, num_workers = 4)
  test_loader = torch.utils.data.DataLoader(dataTest, batch_size = batch_size, shuffle = True, num_workers = 4)

  train_iter = iter(train_loader)
  val_iter = iter(val_loader)
  # test_iter = iter(test_loader)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    # training
    train_acc, train_obj = train(train_iter, val_iter, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    # if args.epochs-epoch<=1:
    if epoch % 5 == 0:
      test_acc, test_obj = infer(test_loader, model, criterion)
      logging.info('test_acc %f', test_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_iter, val_iter, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for i in range(total_train_batches):
  # for step, (input, target) in enumerate(train_queue):
    model.train()
    x_support_set, y_support_set, x_target, y_target = next(train_iter)
    x_support_set = Variable(x_support_set).float()
    y_support_set = Variable(y_support_set, requires_grad=False).long()
    x_target = Variable(x_target.squeeze()).float()
    y_target = Variable(y_target.squeeze(), requires_grad=False).long()

    # convert to one hot encoding
    y_support_set = y_support_set.unsqueeze(2)
    sequence_length = y_support_set.size()[1]
    batch_size = y_support_set.size()[0]
    y_support_set_one_hot = Variable(
        torch.zeros(batch_size, sequence_length, classes_per_set).scatter_(2,
                                                                                y_support_set.data,
                                                                                1), requires_grad=False)
    
    n = x_support_set.size(0)
    
    x_support_set_search, y_support_set_search, x_target_search, y_target_search = next(val_iter)
    x_support_set_search = Variable(x_support_set_search).float()
    y_support_set_search = Variable(y_support_set_search, requires_grad=False).long()
    x_target_search = Variable(x_target_search.squeeze()).float()
    y_target_search = Variable(y_target_search.squeeze(), requires_grad=False).long()

    # convert to one hot encoding
    y_support_set_search = y_support_set_search.unsqueeze(2)
    sequence_length_search = y_support_set_search.size()[1]
    batch_size_search = y_support_set_search.size()[0]
    y_support_set_one_hot_search = Variable(
        torch.zeros(batch_size_search, sequence_length_search, classes_per_set).scatter_(2,
                                                                                y_support_set_search.data,
                                                                                1), requires_grad=False)
    
    x_support_set = x_support_set.cuda()
    y_support_set_one_hot = y_support_set_one_hot.cuda()
    x_target = x_target.cuda()
    y_target = y_target.cuda()
    x_support_set_search = x_support_set_search.cuda()
    y_support_set_one_hot_search = y_support_set_one_hot_search.cuda()
    x_target_search = x_target_search.cuda()
    y_target_search = y_target_search.cuda()
    if epoch>=10:
      architect.step(x_support_set,y_support_set_one_hot,x_target,y_target, x_support_set_search,y_support_set_one_hot_search,x_target_search,y_target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(x_support_set,y_support_set_one_hot,x_target,y_target)
    loss = criterion(logits, y_target.long())

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, y_target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if i % args.report_freq == 0:
      logging.info('train %03d %e %f %f', i, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(test_loader, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for i, (x_support_set_search, y_support_set_search, x_target_search, y_target_search) in enumerate(test_loader):
      x_support_set_search = Variable(x_support_set_search).float()
      y_support_set_search = Variable(y_support_set_search, requires_grad=False).long()
      x_target_search = Variable(x_target_search.squeeze()).float()
      y_target_search = Variable(y_target_search.squeeze(), requires_grad=False).long()

      # convert to one hot encoding
      y_support_set_search = y_support_set_search.unsqueeze(2)
      sequence_length_search = y_support_set_search.size()[1]
      batch_size_search = y_support_set_search.size()[0]
      y_support_set_one_hot_search = Variable(
          torch.zeros(batch_size_search, sequence_length_search, classes_per_set).scatter_(2,
                                                                                  y_support_set_search.data,
                                                                                  1), requires_grad=False)

      x_support_set_search = x_support_set_search.cuda()
      y_support_set_one_hot_search = y_support_set_one_hot_search.cuda()
      x_target_search = x_target_search.cuda()
      y_target_search = y_target_search.cuda()

      logits = model(x_support_set_search,y_support_set_one_hot_search,x_target_search,y_target_search)
      loss = criterion(logits, y_target_search.long())

      prec1, prec5 = utils.accuracy(logits, y_target_search, topk=(1, 5))
      n = x_support_set_search.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if i % args.report_freq == 0:
        logging.info('test %03d %e %f %f', i, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def evaluate(data, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for i in range(total_test_batches):
      x_support_set_search, y_support_set_search, x_target_search, y_target_search = data.get_test_batch(False)
      x_support_set_search = Variable(torch.from_numpy(x_support_set_search)).float()
      y_support_set_search = Variable(torch.from_numpy(y_support_set_search), requires_grad=False).long()
      x_target_search = Variable(torch.from_numpy(x_target_search)).float()
      y_target_search = Variable(torch.from_numpy(y_target_search), requires_grad=False).squeeze().long()

      # convert to one hot encoding
      y_support_set_search = y_support_set_search.unsqueeze(2)
      sequence_length_search = y_support_set_search.size()[1]
      batch_size_search = y_support_set_search.size()[0]
      y_support_set_one_hot_search = Variable(
          torch.zeros(batch_size_search, sequence_length_search, classes_per_set).scatter_(2,
                                                                                  y_support_set_search.data,
                                                                                  1), requires_grad=False)

      # reshape channels and change order
      size_search = x_support_set_search.size()
      x_support_set_search = x_support_set_search.permute(0, 1, 4, 2, 3)
      x_target_search = x_target_search.permute(0, 3, 1, 2)

      x_support_set_search = x_support_set_search.cuda()
      y_support_set_one_hot_search = y_support_set_one_hot_search.cuda()
      x_target_search = x_target_search.cuda()
      y_target_search = y_target_search.cuda()

      logits = model(x_support_set_search,y_support_set_one_hot_search,x_target_search,y_target_search)
      loss = criterion(logits, y_target_search.long())

      prec1, prec5 = utils.accuracy(logits, y_target_search, topk=(1, 5))
      n = x_support_set_search.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if i % args.report_freq == 0:
        logging.info('test %03d %e %f %f', i, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

if __name__ == '__main__':
  main() 