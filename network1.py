import torch
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np

data_transform = transforms.Compose([
		#transforms.RandomResizedCrop(224),
		#transforms.RandomHorizontalFlip(),
		transforms.Resize(224),
		transforms.ToTensor()
		#transforms.Normalize(mean=[0.485, 0.456, 0.406],
		#					 std=[0.229, 0.224, 0.225])
	])
hymenoptera_dataset = datasets.ImageFolder(root='./data/train',
										   transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
											 batch_size=64, shuffle=True,
											 num_workers=0)
#cuda = torch.device('cuda') 
test_dataset = datasets.ImageFolder(root='./data/val', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset,
											 batch_size=1, shuffle=True,
											 num_workers=0)


#255 by 255 by 3
#from PIL import Image
#import numpy
#im = Image.open('./data/val/angel/pic200.png')
#print(numpy.array(im).shape)
#exit()


labels_ = ["aircraft carrier", "angel", "birthday cake", "car", "hand", "leg", "purse", "shoe"]

class Net(nn.Module):
	def __init__(self, num_classes):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.act1 = nn.LeakyReLU(0.1)
		self.conv2 = nn.Conv2d(6, 16, 3)
		self.conv3 = nn.Conv2d(16, 16, 3)
		self.act2 = nn.LeakyReLU(0.1)
		self.fc1 = nn.Linear(16 * 26 * 26, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, num_classes)

	def forward(self, x):
		x = self.pool(self.act1(self.conv1(x)))
		x = self.pool(self.act1(self.conv2(x)))
		x = self.pool(self.act1(self.conv3(x)))
		x = x.view(-1, 16 * 26 * 26)
		x = self.act1(self.fc1(x))
		x = self.act1(self.fc2(x))
		#x = self.fc3(x)
		return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#net = Net(len(labels_))
net = torch.load("troll90.pth")
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
images = []
class_ = []
features = []
threshold = 300
for epoch in range(1):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(test_loader, 0):
		#get the inputs; data is a list of [inputs, labels]
		#inputs, labels = data
		inputs, labels = data[0].to(device), data[1].to(device)
		images.append(inputs.cpu().detach().numpy())
			#print(labels.cpu().detach().numpy()[0])
		class_.append(labels.cpu().detach().numpy()[0])
		print(i)
			#print(np.array(images).shape)
		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		outputs = outputs.cpu().detach().numpy() 
		features.append(outputs)
		#print(outputs)
		"""loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 15 == 14:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 15))
			running_loss = 0.0

	correct = 0
	total = 0
	with torch.no_grad():
		for data in test_loader:
			#images, labels = data
			images, labels = data[0].to(device), data[1].to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the test images: %d %%' % (
		100 * correct / total))
	if epoch % 10 == 0:
		torch.save(net, "troll" + str(epoch) + ".pth")"""
print('Finished Training')
features = np.array(features)
features = np.squeeze(features, axis = 1)
np.set_printoptions(suppress=True)


#print(features[0])
#print(features[1])
#print(features[300])

import scipy
from scipy.spatial.distance import pdist, correlation, squareform
Y = pdist(features, 'minkowski', p=2.)
Y = squareform(Y)
# Ensure that the diagonal isn't the minimum
Y += np.amax(Y)*np.identity(len(Y))
print(Y.shape)
closest_ = 0
for i in range(50, 70, 1):
	closest_ = np.argmin(Y[i])
	if closest_ < threshold and Y[i, closest_] < 0.75:
		print(Y[i, closest_])
		print(images[closest_].shape)
		break
first_im = np.swapaxes(np.squeeze(images[i], axis = 0), 0, 2)
print(closest_)
second_im = np.swapaxes(np.squeeze(images[closest_], axis = 0), 0, 2)
print(labels_[class_[i]])
print(labels_[class_[closest_]])

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#print(np.argmin(Y[0]))
#print(np.argmin(Y[1]))
#print(Y[1795:1805])
fig, axs = plt.subplots(1, 2, figsize=(224, 224))
axs[0].imshow(first_im)
axs[1].imshow(second_im)
plt.show()
