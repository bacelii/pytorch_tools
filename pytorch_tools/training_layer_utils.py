'''

How to use Batchnorm ---- done before the activation -----


class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv1=nn.Conv2d(1,32,3,1)
    self.conv1_bn=nn.BatchNorm2d(32)
    
    self.conv2=nn.Conv2d(32,64,3,1)
    self.conv2_bn=nn.BatchNorm2d(64)
    
    self.dropout1=nn.Dropout(0.25)
    
    self.fc1=nn.Linear(9216,128)
    self.fc1_bn=nn.BatchNorm1d(128)
    
    self.fc2=nn.Linear(128,10)
  def forward(self,x):
    x=self.conv1(x)
    x=F.relu(self.conv1_bn(x))
    
    x=self.conv2(x)
    x=F.relu(self.conv2_bn(x))
    
    x=F.max_pool2d(x,2)
    x=self.dropout1(x)
    
    x=torch.flatten(x,1)
    
    x=self.fc1(x)
    x=F.relu(self.fc1_bn(x))
    
    x=self.fc2(x)
    output=F.log_softmax(x,dim=1)
    return output




'''
