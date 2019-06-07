from Matrix import Matrix
import pickle,random

class Network(object):
    
    def __init__(self,number_Of_Inputs,*args):
        self.synapses = []
        self.layers = [number_Of_Inputs+1]
        for i in args:
            self.layers.append(i)
        for i in range(len(self.layers)-1):
            self.synapses.append(Matrix(height=self.layers[i],width=self.layers[i+1]))
        
    def add_Hidden_Layer(self,number_Of_Neurons):
        self.layers.append(number_Of_Neurons)
        
    def add_Output_Layer(self,number_Of_Outputs):
        self.layers.append(number_Of_Outputs)
        
    def mini_Batch(self,input_Data,output_Data,batch_Size=25):
        if not(batch_Size):
            return((Matrix(input_Data),Matrix(output_Data)))
        a,batch_Input_Data,batch_Output_Data = [],[],[]
        for i in range(batch_Size):
            a.append(random.randint(0,len(input_Data)-1))
        for i in a:
            batch_Input_Data.append(input_Data[i])
            batch_Output_Data.append(output_Data[i])
        return((Matrix(batch_Input_Data),Matrix(batch_Output_Data)))

    def forward_Pass(self,input_Matrix):
        layers = [input_Matrix]
        for j in range(len(self.layers)-1):
            if j != len(self.layers)-1:
                layers.append(layers[j].dot(self.synapses[j]).sigmoid())
                if j!=0:
                    layers[j].add_Weights()
            else:
                layers.append(layers[j].dot(self.synapses[j]).softmax())
        return(layers)
        
    def train(self,input_Data,output_Data,number_Of_Epochs=100,batch_Size=25):
        input_Matrix,output_Matrix = self.mini_Batch(input_Data,output_Data,batch_Size)
        for i in range(number_Of_Epochs):
            input_Matrix,output_Matrix = self.mini_Batch(input_Data,output_Data,batch_Size)
            layers = self.forward_Pass(input_Matrix)
            output_Error = output_Matrix - layers[-1]
            deltas = [output_Error*layers[-1].sigmoid(True)]
            for j in range(len(self.layers)-2):
                synapse_Index = len(self.synapses)-(1+j)
                synapse = self.synapses[synapse_Index].transpose()
                layer = layers[-(j+2)].sigmoid(True)
                delta = deltas[j]
                deltas.append(delta.dot(synapse)*layer)
            for j in range(len(self.synapses)):
                self.synapses[j] += layers[j].transpose().dot(deltas[-(j+1)])
            
    def predict(self,input_Data):
        return(self.forward_Pass(input_Data)[-1].one_Hot())
    
    def save(self,file_Name):
        pickle.dump(self.synapses,open(file_Name,"wb"))
        
    def load(self,file_Name):
        self.synapses = pickle.load(open(file_Name,"rb"))
        
    def test(self,images,labels,batch_Size=False):
        input_Matrix,expected_Output_Matrix = self.mini_Batch(images,labels,batch_Size)
        output_Matrix = self.predict(input_Matrix)
        correct,incorrect = 0,0
        for i in range(output_Matrix.height):
            if output_Matrix.array[i] == expected_Output_Matrix.array[i]:
                correct += 1
            else:
                incorrect += 1
        print("number of correct predictions: " + str(correct))
        print("number of incorrect predictions: " + str(incorrect))
        print("percent correctly classified: " + str((correct/output_Matrix.height)*100)+"%")
        return((correct/output_Matrix.height)*100)
    
