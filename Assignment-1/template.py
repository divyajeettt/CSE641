
# Class structure for Architecture, Activation function and Loss Function.
# For Training, loss function and optimizer you can use code provided in lecture-3 notebook

class Linear:
	def __init__(self,):
		#TODO Define Linear layers here.

	def forward(self,):
		#TODO Write forward logic here.
        
    def backward(self,):
    	#TODO Write backward logic here.
    	
    	
class Activation:
	def forward(self,):
		#TODO: Write Forward logic here.

    def backward(self,):
		#TODO: Write Backward logic here.

class Loss:
	def forward(self,):
    	#TODO: Write forward logic here
	def backward(self):
    	#TODO: Write backward logic here
    	
class MNISTDataset:
	def __init__(self):
		#TODO: Define data, target and transform here.
	
	def __len__(self,):
		return #TODO: Return total length of data
	
	def __getitem__(self,):
		return #TODO: Return each sample in dataset with it transforms applied with label.