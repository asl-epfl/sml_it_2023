from parameters import *
import torch
from torchvision import datasets, transforms


def balance_two_datasets(train_0, train_1):
    '''
    Balance two datasets.
    
    Parameters
    ----------
    train_0: FloatTensor
        dataset 1
    train_1: FloatTensor
        dataset 2
        
    Returns
    -------
    train_0: FloatTensor
        balanced dataset 1
    train_1: FloatTensor
        balanced dataset 2
    '''

    d_train = len(train_1) - len(train_0)
    if d_train > 0:
        train_1 = train_1[:-d_train]
    elif d_train < 0:
        train_0 = train_0[:d_train]
    return train_0, train_1

def split_dataset(dataset, num_agents_sqr):
    '''
    Split dataset dimensions among agents.
    
    Parameters
    ----------
    dataset: Dataset
        dataset
    num_agents_sqr: int
        square root of number of agents
    '''
    
    dataset.data = [list(torch.tensor_split(x, num_agents_sqr, 2)) for x in torch.tensor_split(dataset.data, num_agents_sqr, 1)]

def reduce_to_classes(dataset, bin_classes, train_size=-2):
    '''
    Reduce the dataset to 1 or 2 classes.
    
    Parameters
    ----------
    dataset: Dataset
        dataset
    bin_classes: list(int)
        list of classes
    train_size: int
        number of training samples
    '''
    
    if len(bin_classes) == 2:
        mask_0 = dataset.targets == bin_classes[0]
        mask_1 = dataset.targets == bin_classes[1]

        if train_size > 0:

            idx0 = torch.randint(0, sum(mask_0).item(), (train_size//2,))
            idx1 = torch.randint(0, sum(mask_1).item(), (train_size//2,))

            targets_0 = dataset.targets[mask_0][idx0]
            targets_1 = dataset.targets[mask_1][idx1]
            data_0 = dataset.data[mask_0][idx0]
            data_1 = dataset.data[mask_1][idx1]
        else:
            targets_0 = dataset.targets[mask_0]
            targets_1 = dataset.targets[mask_1]
            data_0 = dataset.data[mask_0]
            data_1 = dataset.data[mask_1]

        dataset.targets = torch.cat((targets_0, targets_1), 0)
        dataset.data = torch.cat((data_0, data_1), 0)
    else:
        mask = (dataset.targets == bin_classes[0])
        dataset.targets = dataset.targets[mask]
        dataset.data = dataset.data[mask]



def generate_bin_MNIST(batch_size, num_agents_sqr, classes = [0, 1], train_size=-2):
    '''
    Generate balanced binary MNIST training/ testing Datasets and Dataloaders for all agents.
    
    Parameters
    ----------
    batch_size: int
        size of batches
    num_agents_sqr: int
        square root of number of agents
    classes: list(int)
        list of MNIST classes 
    train_size: int
        number of training samples
    
    Returns
    -------
    trainloader: DataLoader
        training DataLoader
    testloader: DataLoader
        test DataLoader
    trainset: Dataset
        training Dataset
    testset: Dataset
        test Dataset
    '''
    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
    
    testset = datasets.MNIST('mnist_train', 
                             download=True, 
                             train=True, 
                             transform=transform)
    trainset = datasets.MNIST('mnist_test', 
                              download=True, 
                              train=False, 
                              transform=transform)
    
    reduce_to_classes(trainset, 
                      classes, 
                      train_size)
    reduce_to_classes(testset, classes)

    split_dataset(trainset, num_agents_sqr)
    split_dataset(testset, num_agents_sqr)

    ids1 = (testset.targets == 1).nonzero(as_tuple=True)[0]
    ids0 = (testset.targets == 0).nonzero(as_tuple=True)[0]

    ids = []
    for k in range(N_CYCLES):
        if k % 2 == 0:
            ids.append(ids0[k//2 * N_TEST_CYCLE:(k//2 + 1) * N_TEST_CYCLE])
        else:
            ids.append(ids1[(k - 1)//2 * N_TEST_CYCLE : (k + 1)//2 * N_TEST_CYCLE])
    ids = torch.cat(ids)
    
    trainloader, testloader = [], []
    for i in range(num_agents_sqr):
        for j in range(num_agents_sqr):
            testset.data[i][j] = testset.data[i][j][ids]
            train_ds = torch.utils.data.TensorDataset(trainset.data[i][j], trainset.targets)
            train_ds = MyDataset(train_ds)
            test_ds = torch.utils.data.TensorDataset(testset.data[i][j], testset.targets[ids])
            trainloader.append(torch.utils.data.DataLoader(train_ds, 
                                                           batch_size=batch_size, 
                                                           shuffle=True))
            testloader.append(torch.utils.data.DataLoader(test_ds, 
                                                          batch_size=1, 
                                                          shuffle=False))

    return trainloader, testloader, trainset, testset



class MyDataset(torch.utils.data.Dataset):
    '''
    Creates a customized dataset.
    
    Attributes
    ----------
    ds :  Dataset
        training dataset

    '''
    def __init__(self, ds):
        self.dataset = ds

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

    
def generate_test_shuffled(num_agents_sqr, size, classes = [0]):
    '''
    Generate shuffled prediction dataset.
    
    Parameters
    ----------
    num_agents_sqr: int
        square root of number of agents
    size: int
        number of prediction samples
    classes: list(int)
        list of MNIST classes 
    
    Returns
    -------
    testloader: DataLoader
        test loader
    testset: Dataset
        test dataset
    '''
    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
    
    testset = datasets.MNIST('mnist_train', 
                             download=True, 
                             train=True, 
                             transform=transform)
    
    reduce_to_classes(testset, classes)

    randidx = torch.randint(0, len(testset.targets), (size,))
    testset.targets = testset.targets[randidx]
    testset.data = testset.data[randidx]

    split_dataset(testset, num_agents_sqr)

    testset.targets, ids = testset.targets.sort()
    testloader = []
    for i in range(num_agents_sqr):
        for j in range(num_agents_sqr):
            testset.data[i][j] = testset.data[i][j][ids]

            test_ds = torch.utils.data.TensorDataset(testset.data[i][j], testset.targets)

            testloader.append(torch.utils.data.DataLoader(test_ds, 
                                                          batch_size=1, 
                                                          shuffle=True))
    return testloader, testset

def generate_test_shuffled_2(num_agents_sqr, size, classes = [0, 1]):
    '''
    Generate shuffled prediction dataset for 2 classes.
    
    Parameters
    ----------
    num_agents_sqr: int
        square root of number of agents
    size: int
        number of prediction samples
    classes: list(int)
        list of MNIST classes 
    
    Returns
    -------
    testloader: DataLoader
        test loader
    testset0: Dataset
        test dataset
    
    '''
    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
    
    testset0 = datasets.MNIST('mnist_train', 
                              download=True, 
                              train=True, 
                              transform=transform)
    testset1 = datasets.MNIST('mnist_train', 
                              download=True, 
                              train=True, 
                              transform=transform)
    
    reduce_to_classes(testset0, [classes[0]])
    reduce_to_classes(testset1, [classes[1]])

    randidx0 = torch.randint(0, len(testset0.targets), (size,))
    randidx1 = torch.randint(0, len(testset1.targets), (size,))

    testset0.targets = testset0.targets[randidx0]
    testset0.data = testset0.data[randidx0]
    testset1.targets = testset1.targets[randidx1]
    testset1.data = testset1.data[randidx1]

    split_dataset(testset0, num_agents_sqr)
    split_dataset(testset1, num_agents_sqr)
    testset0.targets = torch.cat((testset0.targets, testset1.targets), 0)

    testloader = []
    for i in range(num_agents_sqr):
        for j in range(num_agents_sqr):
            testset0.data[i][j] = torch.cat((testset0.data[i][j], testset1.data[i][j]),0)
            test_ds = torch.utils.data.TensorDataset(testset0.data[i][j], testset0.targets)
            testloader.append(torch.utils.data.DataLoader(test_ds, 
                                                          batch_size=1, 
                                                          shuffle=True))
    return testloader, testset0


def generate_multi_MNIST(batch_size, num_agents_sqr, classes = [0, 1], train_size=-2):
    '''
    Generate balanced multi-class MNIST training/ testing Datasets and Dataloaders for all agents.
    
    Parameters
    ----------
    batch_size: int
        size of batches
    num_agents_sqr: int
        square root of number of agents
    classes: list(int)
        list of MNIST classes 
    train_size: int
        number of training samples
    
    Returns
    -------
    trainloader: DataLoader
        training DataLoader
    testloader: DataLoader
        test DataLoader
    trainset: Dataset
        training Dataset
    testset: Dataset
        test Dataset
    '''
    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
    
    trainset = []
    for c in range(num_multiclasses):
        trainset.append(datasets.MNIST('mnist_test', 
                                       download=True, 
                                       train=False, 
                                       transform=transform))
        reduce_to_classes(trainset[c], [c])
        randidx = torch.randint(0, len(trainset[c].targets), (train_size // num_multiclasses,))
        trainset[c].targets = trainset[c].targets[randidx]
        trainset[c].data = trainset[c].data[randidx]
        split_dataset(trainset[c], num_agents_sqr)

    train_targets = torch.cat([trainset[c].targets for c in range(num_multiclasses)], 0)
    testset = datasets.MNIST('mnist_train', 
                             download=True, 
                             train=True, 
                             transform=transform)
    randidx_test = torch.randint(0, len(testset.targets), (N_TEST_MULTI * num_multiclasses,))
    testset.targets = testset.targets[randidx_test]
    testset.data = testset.data[randidx_test]
    split_dataset(testset, num_agents_sqr)

    testset.targets, ids = testset.targets.sort()

    trainloader, testloader = [], []
    for i in range(num_agents_sqr):
        for j in range(num_agents_sqr):
            testset.data[i][j] = testset.data[i][j][ids]
            train_ds = torch.utils.data.TensorDataset(torch.cat([trainset[c].data[i][j] for c in range(num_multiclasses)], 0), train_targets)
            train_ds = MyDataset(train_ds)
            test_ds = torch.utils.data.TensorDataset(testset.data[i][j], testset.targets)
            
            trainloader.append(torch.utils.data.DataLoader(train_ds, 
                                                           batch_size=batch_size, 
                                                           shuffle=True))
            testloader.append(torch.utils.data.DataLoader(test_ds, 
                                                          batch_size=1, 
                                                          shuffle=False))
    return trainloader, testloader, trainset, testset


def generate_Gaussian(batch_size, N_TRAIN, N_TEST):
    '''
    Generate balanced binary training/ testing Datasets and Dataloaders for the Gaussian example.
    
    Parameters
    ----------
    batch_size: int
        size of batches
    N_TRAIN: int
        number of training samples
    N_TEST: int
        number of test samples
    
    Returns
    -------
    trainloader: DataLoader
        training DataLoader
    testloader: DataLoader
        test DataLoader
    train_ds: Dataset
        training Dataset
    test_ds: Dataset
        test Dataset
    '''
    
    train_set = Q1.rvs((num_agents, 
                        num_classes, 
                        N_TRAIN // num_classes))
    train_set[1, 1, :, :] = Q2.rvs(N_TRAIN // num_classes)
    train_set = torch.Tensor(train_set.reshape((train_set.shape[0], train_set.shape[1] * train_set.shape[2], -1)))
    train_targets = torch.cat((torch.zeros(N_TRAIN // num_classes), torch.ones(N_TRAIN // num_classes)))
    test_set = torch.Tensor(Q1.rvs((num_agents, N_TEST)))
    test_targets = torch.zeros(N_TEST)

    trainloader, testloader = [], []

    for i in range(num_agents):
        train_ds = torch.utils.data.TensorDataset(train_set[i], train_targets)
        train_ds = MyDataset(train_ds)
        test_ds = torch.utils.data.TensorDataset(test_set[i], test_targets)

        trainloader.append(torch.utils.data.DataLoader(train_ds, 
                                                       batch_size=batch_size, 
                                                       shuffle=True))
        testloader.append(torch.utils.data.DataLoader(test_ds, 
                                                      batch_size=1, 
                                                      shuffle=False))
    return trainloader, testloader, train_ds, test_ds
