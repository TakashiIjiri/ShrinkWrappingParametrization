import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image


import glob
import re

# input : 6146頂点のモデル
# 8頂点 * 3次元
# 6138点のoffset値
# 6138 + 24 = 6162

# note nn.Sigmoid は非推奨かも
# 恐らくResNetとかをやるときにはFalse指定する必要ある

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(6162, 1000),
			nn.Sigmoid(),
			nn.Linear(1000, 10),
			nn.Sigmoid())

		self.decoder = nn.Sequential(
			nn.Linear(10, 1000),
			nn.Sigmoid(),
			nn.Linear(1000, 6162))

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


# ------------------------------
# step 0 :　setup parameters---- 
cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available!')

NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EXPORT_DIR    = './autoencoder_sw'


if not os.path.exists(EXPORT_DIR):
    os.mkdir(EXPORT_DIR)



# ------------------------------
# step 1 : load dataset---------
raw_data = []
for fname in glob.glob('./models/*.txt') :
	file = open(fname)
	tmp = file.read()
	tmp = tmp.split()
	tmp = [float(i) for i in tmp]
	del tmp[24:24+8]
	raw_data.append( tmp )
	# print(fname, len(tmp), type(tmp[0]), type(tmp[100]))

# array --> ndarray
raw_data = np.array(raw_data)
print(raw_data.dtype, raw_data.shape) # float64 (1240, 6162)

#ndarray --> tensor
train_data = torch.tensor(raw_data, dtype = torch.float32)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)



# ------------------------------
# step 2 : 訓練-----------------

model = Autoencoder()
model.load_state_dict(torch.load('{}/autoencoder.pth'.format(EXPORT_DIR),
                                 map_location=lambda storage,
                                 loc: storage))

print("--------------------------------------")
print("--------------------------------------")
print("--------------------------------------")
example = torch.rand(1, 6162) # 入力のshapeに合わせる
traced_net = torch.jit.trace(model, example)
traced_net.save("full.pt")

traced_net = torch.jit.trace(model.encoder, example)
traced_net.save("encoder.pt")

example = torch.rand(1, 10) # 入力のshapeに合わせる
traced_net = torch.jit.trace(model.decoder, example)
traced_net.save("decoder.pt")

# うごかしてさをかくにん
z = model.encoder(Variable(train_data, volatile=True)).data.numpy()

print ( z.shape )


