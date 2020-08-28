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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import Networkminiimagenet as Network

from miniImagenet import miniImagenetOneShotDataset

parser = argparse.ArgumentParser("miniImagenet")
parser.add_argument('--set', type=str, default='miniImagenet', help='dataset name')
parser.add_argument('--batch_size', type=int, default=60, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='miniImagenet_5way_1shot', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--retrain', default=None, help='whether using retrain weights')
parser.add_argument('--retrain', default='eval-EXP-20200729-192400/weights.pt', help='whether using retrain weights')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

batch_size = args.batch_size
fce = True
classes_per_set = 5
samples_per_class = 1
channels = 1

# total_epochs = 200
total_train_batches = 500
total_val_batches = 200
total_test_batches = 500
best_val_acc = 0.0
dataroot = 'miniimagenet'

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

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, classes_per_set, args.layers, args.auxiliary, genotype, args.batch_size)
  model = model.cuda()

  if args.retrain:
    utils.load(model, args.retrain)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar10(args)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc = 0.0
  best_test_acc = 0.0

  dataTrain = miniImagenetOneShotDataset(dataroot = dataroot, 
                                          type = 'train', 
                                          nEpisodes = total_train_batches * batch_size * args.epochs, 
                                          classes_per_set = classes_per_set, 
                                          samples_per_class = samples_per_class)
  dataVal = miniImagenetOneShotDataset(dataroot = dataroot, 
                                          type = 'val', 
                                          nEpisodes = total_val_batches * batch_size * args.epochs, 
                                          classes_per_set = classes_per_set, 
                                          samples_per_class = samples_per_class)
  dataTest = miniImagenetOneShotDataset(dataroot = dataroot, 
                                          type = 'test', 
                                          nEpisodes = total_test_batches * batch_size, 
                                          classes_per_set = classes_per_set, 
                                          samples_per_class = samples_per_class)
  train_loader = torch.utils.data.DataLoader(dataTrain, batch_size = batch_size, shuffle = True, num_workers = 4)
  val_loader = torch.utils.data.DataLoader(dataVal, batch_size = batch_size, shuffle = True, num_workers = 4)
  test_loader = torch.utils.data.DataLoader(dataTest, batch_size = batch_size, shuffle = True, num_workers = 4)

  train_iter = iter(train_loader)
  val_iter = iter(val_loader)
  test_iter = iter(test_loader)

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_iter, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(val_iter, model, criterion)
    test_acc, test_obj = evaluate(test_loader, model, criterion)
    logging.info('test_acc %f', test_acc)
    best_test_acc = max(best_test_acc, test_acc)

    if valid_acc > best_acc:
      logging.info('final_test_acc %f', test_acc)
    
    best_acc = max(valid_acc, best_acc)
    logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)
    logging.info('test_acc %f, best_test_acc %f', test_acc, best_test_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

def train(train_iter, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

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
    x_support_set = x_support_set.cuda()
    y_support_set_one_hot = y_support_set_one_hot.cuda()
    x_target = x_target.cuda()
    y_target = y_target.cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(x_support_set,y_support_set_one_hot,x_target,y_target)
    loss = criterion(logits, y_target.long())
    if args.auxiliary:
      loss_aux = criterion(logits_aux, y_target.long())
      loss += args.auxiliary_weight*loss_aux
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

def infer(val_iter, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for i in range(total_val_batches):
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
      x_support_set_search = x_support_set_search.cuda()
      y_support_set_one_hot_search = y_support_set_one_hot_search.cuda()
      x_target_search = x_target_search.cuda()
      y_target_search = y_target_search.cuda()

      logits, _ = model(x_support_set_search,y_support_set_one_hot_search,x_target_search,y_target_search)
      loss = criterion(logits, y_target_search.long())

      prec1, prec5 = utils.accuracy(logits, y_target_search, topk=(1, 5))
      n = x_support_set_search.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if i % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', i, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def evaluate(test_loader, model, criterion):
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

      logits, _ = model(x_support_set_search,y_support_set_one_hot_search,x_target_search,y_target_search)
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