dataset = "../../Datasets/Kvasir - Aziz"
model = 'resnet18'
batch_size = 64 # help='input batch size for training (default: 128)')
epochs = 2 # help='number of epochs to train (default: 20)')
learning_rate = 0.1 # help='learning rate')
data_augmentation = True # 'augment data by flipping and cropping'
cutout = True # help='apply cutout')
n_holes = 1 #help='number of holes to cut out from image')
length = 16 # help='length of the holes')
no_cuda = True # help='enables CUDA training')
seed = 0 # help='random seed (default: 1)')
iterations = 3 # type=int, help='Number of experiments to run'
