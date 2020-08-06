from titanic.passdata import passdata
from titanic.model import model

obj = passdata()
train,test = obj.preprocess()
print(train.head(10))
model = model(train,test)