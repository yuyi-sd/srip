import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path

import math
import numpy as np

class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class NetworkOmniglot(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, batch_size):
    super(NetworkOmniglot, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self._batch_size = batch_size

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(1, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
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
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # self.classifier = nn.Linear(C_prev, num_classes)

    self.lstm = BidirectionalLSTM(layer_size=[32],batch_size=self._batch_size,vector_dim = C_prev,use_cuda=True)
    self.dn = DistanceNetwork()
    self.classify = AttentionalClassify()


  def forward(self, support_set_images, support_set_y_one_hot, target_image, target_y):
    encoded_images = []
    for i in np.arange(support_set_images.size(1)):
      input = support_set_images[:, i, :, :]
      s0 = s1 = self.stem(input)
      for i, cell in enumerate(self.cells):
        s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      out = self.global_pooling(s1)
      gen_encode = out.view(out.size(0),-1)
      encoded_images.append(gen_encode)

    # produce embeddings for target images
    input = target_image
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
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
    return logits, logits_aux

class Networkminiimagenet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, batch_size):
    super(Networkminiimagenet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self._batch_size = batch_size

    stem_multiplier = 3
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
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    # self.classifier = nn.Linear(C_prev, num_classes)

    self.lstm = BidirectionalLSTM(layer_size=[32],batch_size=self._batch_size,vector_dim = C_prev,use_cuda=True)
    self.dn = DistanceNetwork()
    self.classify = AttentionalClassify()


  def forward(self, support_set_images, support_set_y_one_hot, target_image, target_y):
    encoded_images = []
    for i in np.arange(support_set_images.size(1)):
      input = support_set_images[:, i, :, :]
      s0 = self.stem0(input)
      s1 = self.stem1(s0)
      for i, cell in enumerate(self.cells):
        s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      out = self.global_pooling(s1)
      gen_encode = out.view(out.size(0),-1)
      encoded_images.append(gen_encode)

    # produce embeddings for target images
    input = target_image
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
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
    return logits, logits_aux

class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
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
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux

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