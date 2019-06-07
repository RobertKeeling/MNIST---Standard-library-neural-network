import csv,tkinter,pickle
from Network import Network
from Matrix import Matrix

def hexrgb(r,g,b):
    rgb = "#"
    for i in [r,g,b]:
        x = str(hex(i))[2:]
        if len(x)==1:
            x = "00" + x
        elif len(x)==2:
            x = "0" + x
        rgb += x
    return(rgb)

class GUI:
    
    def __init__(self):
        self.counter,self.correct,self.incorrect = 0,0,0
        self.window = tkinter.Tk()
        self.window.geometry("290x360")
        self.can = tkinter.Canvas(self.window)
        self.can.place(relwidth=1,relheight=1,relx=0,rely=0)
        self.window.bind_all("<n>",self.draw)
        self.draw()
        self.window.mainloop()
    
    def draw(self,key=0):
        self.can.delete("all")
        x,y = 5,75
        prediction,actual = str(self.predict()),str(predict_Dictionary[str(test_Labels[self.counter])])
        if prediction == actual:
            self.correct += 1
        else:
            self.incorrect += 1
        for a,b,c in [[60,13,"Actual Value: "+actual],
                      [220,13,"Predicted Value: "+prediction],
                      [60,33,"Correct Predictions: "+str(self.correct)],
                      [220,33,"Incorrect Predictions: "+str(self.incorrect)],
                      [145,53,"Percent Correctly Clasified: "+str(int(self.correct/(self.correct+self.incorrect)*100))+"%"]]:
            self.can.create_text(a,b,text=c)
        for i in images[self.counter]:
            i = 255-int(i)
            self.can.create_rectangle(x,y,x+10,y+10,outline=hexrgb(i*16,i*16,i*16),fill=hexrgb(i*16,i*16,i*16))
            x += 10
            if x == 285:
                x,y = 5,y+10
        self.counter += 1

    def predict(self):
        return(predict_Dictionary[str(net.predict(Matrix([images[self.counter]])).array[0])])

predict_Dictionary = {"[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]":0,"[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]":1,
                      "[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]":2,"[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]":3,
                      "[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]":4,"[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]":5,
                      "[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]":6,"[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]":7,
                      "[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]":8,"[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]":9}



images = pickle.load(open("test_Images.pickle","rb"))
test_Labels = pickle.load(open("test_Labels.pickle","rb"))

net = Network(28*28,100,50,10)
net.load("synapses.pickle")

GUI()
