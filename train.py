import pickle
from Network import Network

images = pickle.load(open("training_Images.pickle","rb"))
labels = pickle.load(open("training_Labels.pickle","rb"))
new_Images = pickle.load(open("new_Images.pickle","rb"))
new_Labels = pickle.load(open("new_Labels.pickle","rb"))
test_Images = pickle.load(open("test_Images.pickle","rb"))
test_Labels = pickle.load(open("test_Labels.pickle","rb"))

images += new_Images
labels += new_Labels

net = Network(28*28,120,70,10)

while True:
    net.train(images,labels,100,20)
    score = net.test(test_Images[:1000],test_Labels[:1000])
    net.save("synapses"+","+str(score)+".pickle")

