import os

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

  def __init__(self, use_gpu=False):
    import torch
    if use_gpu:
        self.model = deepsea.cuda()
    else:
        self.model = deepsea
    self.use_gpu = use_gpu
    self.model.load_state_dict(torch.load(RESULT_DIR + '/models-best/deepsea.pth'))

  def predict(self, X, batch_size=256):
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
      preds[b*batch_size:(b+1)*batch_size] = output.data.cpu().numpy()
    if (b+1) * batch_size < X.shape[0]:
      leftover = X[(b+1)*batch_size:]
      # print(leftover.shape[0])
      t = torch.from_numpy(leftover.reshape((leftover.shape[0],4,1,1000))).float()
      inputs = Variable(t, volatile=True) # http://pytorch.org/docs/master/notes/autograd.html
      if self.use_gpu:
        inputs = inputs.cuda()
      output = self.model(inputs)
      preds[(b+1)*batch_size:] = output.data.cpu().numpy()
    return preds

# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/8
class SelectiveSequential(nn.Module):
  def __init__(self, to_select, modules_dict):
    super(SelectiveSequential, self).__init__()
    for key, module in modules_dict.items():
        self.add_module(key, module)
    self._to_select = to_select
  
  def forward(x):
    list = []
    for name, module in self._modules.iteritems():
        x = module(x)
        if name in self._to_select:
            list.append(x)
    return list


class SelectiveDeepSea(nn.Module):
  # features should be taken after the nonlinearity https://www.quora.com/While-using-Convolutional-Neural-Networks-as-feature-extractor-do-we-extract-a-layers-features-after-ReLU-activation-function-or-before
  def __init__(self):
    super().__init__()
    self.features = SelectiveSequential(
        ['pred', 'pool1', 'pool2', 'rel3', 'reldense'],
        {'conv1': nn.Conv2d(4,320,(1, 8)),
         'rel1': nn.Threshold(0,1e-06),
         'pool1': nn.MaxPool2d((1, 4),(1, 4)),
         'drop1': nn.Dropout(0.2),
         'conv2': nn.Conv2d(320,480,(1, 8)),
         'rel2': nn.Threshold(0,1e-06),
         'pool2': nn.MaxPool2d((1, 4),(1, 4)),
         'drop2': nn.Dropout(0.2),
         'conv3': nn.Conv2d(480,960,(1, 8)),
         'rel3': nn.Threshold(0,1e-06),
         'drop3': nn.Dropout(0.5),
         'flatten': Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
         'dense1': nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(50880,925)), # Linear,
         'reldense': nn.Threshold(0,1e-06),
         'dense2': nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(925,919)), # Linear,
         'pred': nn.Sigmoid()}
    )

  def forward(self, x):
    return self.features(x)