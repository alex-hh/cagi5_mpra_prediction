import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import numpy as np

from functools import reduce
from torch.autograd import Variable

RESULT_DIR = os.environ.get('RESULT_DIR', 'data/remote_results')


class LambdaBase(nn.Sequential):
  def __init__(self, fn, *args):
    super(LambdaBase, self).__init__(*args)
    self.lambda_func = fn

  def forward_prepare(self, input):
    output = []
    for module in self._modules.values():
        output.append(module(input))
    return output if output else input

class Lambda(LambdaBase):
  def forward(self, input):
      return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
  def forward(self, input):
      return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
  def forward(self, input):
      return reduce(self.lambda_func,self.forward_prepare(input))

# extracting intermediate outputs: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/7
# could also use 'register forward hook': http://pytorch.org/docs/master/nn.html
deepsea = nn.Sequential( # Sequential,
    nn.Conv2d(4,320,(1, 8)),
    nn.Threshold(0,1e-06),
    nn.MaxPool2d((1, 4),(1, 4)),
    nn.Dropout(0.2),
    nn.Conv2d(320,480,(1, 8)),
    nn.Threshold(0,1e-06),
    nn.MaxPool2d((1, 4),(1, 4)),
    nn.Dropout(0.2),
    nn.Conv2d(480,960,(1, 8)),
    nn.Threshold(0,1e-06),
    nn.Dropout(0.5),
    Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
    nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(50880,925)), # Linear,
    nn.Threshold(0,1e-06),
    nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(925,919)), # Linear,
    nn.Sigmoid(),
)

class DeepSea:

  def __init__(self, use_gpu=False, features=['15']):
    """
    available feature sets: 14: pred, 2: pool1, 6: pool2, 9: conv3, 13: hidden
    """
    import torch
    if use_gpu:
        #self.model = deepsea.cuda()
        self.model = SelectiveDeepSea(features, use_gpu=True).cuda()
    else:
        #self.model = deepsea
        self.model = SelectiveDeepSea(features)
    self.use_gpu = use_gpu
    pretrained_state_dict = torch.load(RESULT_DIR + '/models-best/deepsea.pth')
    pretrained_state_dict = {'features.' + k: v for k,v in pretrained_state_dict.items()} # rename to match selectivedeepsea namespace below
    self.model.load_state_dict(pretrained_state_dict)
    self.features = features

  def layer_activations(self, X, batch_size=256):
    self.model.eval()
    if X.shape[1] == 1000 and X.shape[2] == 4:
      X = X.swapaxes(1,2)
    outputs = [[] for f in self.features]
    for b in range(X.shape[0] // batch_size):
      x = X[b*batch_size:(b+1)*batch_size]  
      t = torch.from_numpy(x.reshape((x.shape[0],4,1,1000))).float()
      inputs = Variable(t, volatile=True)
      if self.use_gpu:
        inputs = inputs.cuda() # http://pytorch.org/docs/master/notes/autograd.html
      output = self.model(inputs)
      for i, a in enumerate(output):
        outputs[i].append(a.data.cpu().numpy())
      
    if (b+1) * batch_size < X.shape[0]:
      leftover = X[(b+1)*batch_size:]
      # print(leftover.shape[0])
      t = torch.from_numpy(leftover.reshape((leftover.shape[0],4,1,1000))).float()
      inputs = Variable(t, volatile=True) # http://pytorch.org/docs/master/notes/autograd.html
      if self.use_gpu:
        inputs = inputs.cuda()
      output = self.model(inputs)
      for i, a in enumerate(output):
        outputs[i].append(a.data.cpu().numpy())
      
    return [np.squeeze(np.concatenate(o, axis=0)) for o in outputs]

  def predict(self, X, batch_size=256):
    assert '15' in self.features
    self.model.eval()
    if X.shape[1] == 1000 and X.shape[2] == 4:
      X = X.swapaxes(1,2)
    preds = np.zeros((X.shape[0],919))
    for b in range(X.shape[0] // batch_size):
      x = X[b*batch_size:(b+1)*batch_size]  
      t = torch.from_numpy(x.reshape((x.shape[0],4,1,1000))).float()
      inputs = Variable(t, volatile=True)
      if self.use_gpu:
        inputs = inputs.cuda() # http://pytorch.org/docs/master/notes/autograd.html
      output = self.model(inputs)
      if type(output)==list:
        output = output[self.features.index('15')]
      preds[b*batch_size:(b+1)*batch_size] = output.data.cpu().numpy()
    if (b+1) * batch_size < X.shape[0]:
      leftover = X[(b+1)*batch_size:]
      # print(leftover.shape[0])
      t = torch.from_numpy(leftover.reshape((leftover.shape[0],4,1,1000))).float()
      inputs = Variable(t, volatile=True) # http://pytorch.org/docs/master/notes/autograd.html
      if self.use_gpu:
        inputs = inputs.cuda()
      output = self.model(inputs)
      print(type(output))
      if type(output)==list:
        output = output[self.features.index('15')].data.cpu().numpy()
      else:
        output = output.data.cpu().numpy()
      preds[(b+1)*batch_size:] = output
    return preds

# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/8
class SelectiveSequential(nn.Module):
  def __init__(self, to_select, modules_list):
    super(SelectiveSequential, self).__init__()
    for key, module in modules_list:
        self.add_module(key, module)
    self._to_select = to_select
  
  def forward(self, x):
    list = []
    for name, module in self._modules.items():
        x = module(x)
        if name in self._to_select:
            list.append(x)
    return list


class SelectiveDeepSea(nn.Module):
  # features should be taken after the nonlinearity https://www.quora.com/While-using-Convolutional-Neural-Networks-as-feature-extractor-do-we-extract-a-layers-features-after-ReLU-activation-function-or-before
  def __init__(self, features, use_gpu=False):
    super().__init__()
    self.features = SelectiveSequential(
        features,
        [('0', nn.Conv2d(4,320,(1, 8))),
         ('1', nn.Threshold(0,1e-06)),
         ('2', nn.MaxPool2d((1, 4),(1, 4))),
         ('3', nn.Dropout(0.2)),
         ('4', nn.Conv2d(320,480,(1, 8))),
         ('5', nn.Threshold(0,1e-06)),
         ('6', nn.MaxPool2d((1, 4),(1, 4))),
         ('7', nn.Dropout(0.2)),
         ('8', nn.Conv2d(480,960,(1, 8))),
         ('9', nn.Threshold(0,1e-06)),
         ('10', nn.Dropout(0.5)),
         ('11', Lambda(lambda x: x.view(x.size(0),-1))), # Reshape,
         ('12', nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(50880,925))), # Linear,
         ('13', nn.Threshold(0,1e-06)),
         ('14', nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(925,919))), # Linear,
         ('15', nn.Sigmoid())])
    if use_gpu:
      self.features = self.features.cuda()
    

  def forward(self, x):
    return self.features(x)
