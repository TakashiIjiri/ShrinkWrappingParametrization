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
if cuda:
    model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

loss_list = []


for epoch in range(NUM_EPOCHS):
	print("----------------", epoch, "------------------")

	for x in train_loader:
		
		# print(type(x), x.shape) # bachsize  * 6162

		if cuda:
			x = Variable(x).cuda()
		else:
			x = Variable(x)

		# forward 
		xhat = model(x)
		# loss
		loss = criterion(xhat, x)

		# optimization 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		loss_list.append(loss.data)

	# ログ
	if epoch % 10 == 0:
		print('epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, NUM_EPOCHS, loss.data))
		torch.save(model.state_dict(), './{}/autoencoder{}.pth'.format(EXPORT_DIR, epoch))

np.save('./{}/loss_list.npy'.format(EXPORT_DIR), np.array(loss_list))
torch.save(model.state_dict(), './{}/autoencoder.pth'.format(EXPORT_DIR))
