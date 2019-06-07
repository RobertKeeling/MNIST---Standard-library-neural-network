import pickle,random
from Matrix import Matrix
from tkinter import *

def cap_colours(x):
    if x < 0:
        x = 0
    elif x > 255:
        x = 255
    return(x)

def create_noise():
    array = []
    for i in range(28):
        for j in range(28):
            array.append(random.randint(-50,50))
    array.append(random.randint(-50,50))
    return(array)

images = pickle.load(open("training_Images.pickle","rb"))
labels = pickle.load(open("training_Labels.pickle","rb"))
image_matrices = []
for image in images:
    image_matrices.append(Matrix([image]))

new_images = []
new_labels = []
for i in range(len(image_matrices)):
    new_images.append((image_matrices[i]+Matrix([create_noise()])).array[0])
    new_labels.append(labels[i])

for image in new_images:
    for i in range(len(image)):
        image[i] = cap_colours(image[i])

pickle.dump(new_images,open("new_images.pickle","wb"))
pickle.dump(new_labels,open("new_labels.pickle","wb"))
















