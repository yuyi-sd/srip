import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import math
import numpy as np

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)
    #self.bn=nn.BatchNorm2d(C//4, affine=False)
    #self.conv1 = nn.Conv2d(C//4,C//4,kernel_size=1,stride=1,padding=0,bias=False)
    for primitive in PRIMITIVES:
      op = OPS[primitive](C//2, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C//2, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//2, :, :]
    xtemp2 = x[ : ,  dim_2//2:, :, :]
    xtemp3 = x[:,dim_2// 4:dim_2// 2, :, :]
    xtemp4 = x[:,dim_2// 2:, :, :]
    
    temp1 = sum(w.to(xtemp.device) * op(xtemp) for w, op in zip(weights, self._ops))
    if temp1.shape[2] == x.shape[2]:
      #ans = torch.cat([temp1,self.bn(self.conv1(xtemp3))],dim=1)
      #ans = torch.cat([ans,xtemp4],dim=1)
      ans = torch.cat([temp1,xtemp2],dim=1)
      #ans = torch.cat([ans,x[:, 2*dim_2// 4: , :, :]],dim=1)
    else:
      #ans = torch.cat([temp1,self.bn(self.conv1(self.mp(xtemp3)))],dim=1)
      #ans = torch.cat([ans,self.mp(xtemp4)],dim=1)

      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)

    ans = channel_shuffle(ans,2)
    return ans


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights,weights2):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(weights2[offset+j].to(self._ops[offset+j](h, weights[offset+j]).device)*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      #s = channel_shuffle(s,4)
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, batch_size, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._batch_size = batch_size

    C_curr = stem_multiplier*C

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr, C_curr, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self.lstm = BidirectionalLSTM(layer_size=[32],batch_size=self._batch_size,vector_dim = C_prev,use_cuda=True)
    self.dn = DistanceNetwork()
    self.classify = AttentionalClassify()
    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, support_set_images, support_set_y_one_hot, target_image, target_y):
    encoded_images = []
    for i in np.arange(support_set_images.size(1)):
      input = support_set_images[:, i, :, :]
      s0 = self.stem0(input)
      s1 = self.stem1(s0)
      for i, cell in enumerate(self.cells):
        if cell.reduction:
          weights = F.softmax(self.alphas_reduce, dim=-1)
          n = 3
          start = 2
          weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
          for i in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            start = end
            n += 1
            weights2 = torch.cat([weights2,tw2],dim=0)
        else:
          weights = F.softmax(self.alphas_normal, dim=-1)
          n = 3
          start = 2
          weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
          for i in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weights2 = torch.cat([weights2,tw2],dim=0)
        s0, s1 = s1, cell(s0, s1, weights,weights2)
      out = self.global_pooling(s1)
      gen_encode = out.view(out.size(0),-1)
      encoded_images.append(gen_encode)

    # produce embeddings for target images
    input = target_image
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      s0, s1 = s1, cell(s0, s1, weights,weights2)
    out = self.global_pooling(s1)
    gen_encode = out.view(out.size(0),-1)
    encoded_images.append(gen_encode)
    output = torch.stack(encoded_images)

    # use fce?
    outputs = self.lstm(output)

    # get similarities between support set embeddings and target
    similarites = self.dn(support_set=output[:-1], input_image=output[-1])

    preds = self.classify(similarites, support_set_y=support_set_y_one_hot)
    logits = preds

    return logits

  def _loss(self, support_set_images, support_set_y_one_hot, target_image, target_y):
    logits = self(support_set_images, support_set_y_one_hot, target_image, target_y)
    return self._criterion(logits, target_y.long()) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.betas_normal = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self.betas_reduce = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
      self.betas_normal,
      self.betas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights,weights2):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
          W[j,:]=W[j,:]*W2[j]
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        
        #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene
    n = 3
    start = 2
    weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
    weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
    for i in range(self._steps-1):
      end = start + n
      tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
      tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
      start = end
      n += 1
      weightsr2 = torch.cat([weightsr2,tw2],dim=0)
      weightsn2 = torch.cat([weightsn2,tn2],dim=0)
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),weightsn2.data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),weightsr2.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

class AttentionalClassify(nn.Module):
  def __init__(self):
    super(AttentionalClassify, self).__init__()

  def forward(self, similarities, support_set_y):
    """
    Products pdfs over the support set classes for the target set image.
    :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
    :param support_set_y:[batch_size,sequence_length,classes_num]
    :return: Softmax pdf shape[batch_size,classes_num]
    """
    softmax = nn.Softmax()
    softmax_similarities = softmax(similarities)
    preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
    return preds


class DistanceNetwork(nn.Module):
  """
  This model calculates the cosine distance between each of the support set embeddings and the target image embeddings.
  """

  def __init__(self):
    super(DistanceNetwork, self).__init__()

  def forward(self, support_set, input_image):
    """
    forward implement
    :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
    :param input_image: the embedding of the target image,shape[batch_size,64]
    :return:shape[batch_size,sequence_length]
    """
    eps = 1e-10
    similarities = []
    for support_image in support_set:
      sum_support = torch.sum(torch.pow(support_image, 2), 1)
      support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
      dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
      cosine_similarity = dot_product * support_manitude
      similarities.append(cosine_similarity)
    similarities = torch.stack(similarities)
    return similarities.t()


class BidirectionalLSTM(nn.Module):
  def __init__(self, layer_size, batch_size, vector_dim,use_cuda):
    super(BidirectionalLSTM, self).__init__()
    """
    Initial a muti-layer Bidirectional LSTM
    :param layer_size: a list of each layer'size
    :param batch_size: 
    :param vector_dim: 
    """
    self.batch_size = batch_size
    self.hidden_size = layer_size[0]
    self.vector_dim = vector_dim
    self.num_layer = len(layer_size)
    self.use_cuda = use_cuda
    self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                        bidirectional=True)
    self.hidden = self.init_hidden(self.use_cuda)

  def init_hidden(self,use_cuda):
    if use_cuda:
      return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda(),
              Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda())
    else:
      return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False),
              Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False))

  def repackage_hidden(self,h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
      return h.detach()
    else:
      return tuple(self.repackage_hidden(v) for v in h)

  def forward(self, inputs):
    # self.hidden = self.init_hidden(self.use_cuda)
    self.hidden = self.repackage_hidden(self.hidden)
    output, self.hidden = self.lstm(inputs, self.hidden)
    return output