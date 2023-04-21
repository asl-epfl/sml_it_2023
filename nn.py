from generate_data import *
import torch
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Module):
    '''
    Creates a sequence of 2 linear layers and Tanh activations.
    
    Attributes
    ----------
    num_feats :  int
        int value indicating input size
    hidden_size : int
        int value indicating hidden layer size
    num_classes : int
        int value indicating output size
        
    Methods
    -------
    forward()
    '''
    
    def __init__(self, num_feats, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_feats, 
                             hidden_size, 
                             bias=True)
        self.ReLU = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, 
                             num_classes, 
                             bias=True)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ReLU(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
    
class Net_Gaussian(nn.Module):
    '''
    Creates a sequence of 3 linear layers and Tanh activations.
    
    Attributes
    ----------
    num_feats :  int
        int value indicating input size
    hidden_size : int
        int value indicating hidden layer size
    num_classes : int
        int value indicating output size
        
    Methods
    -------
    forward()
    '''
    def __init__(self, num_feats, hidden_size, num_classes):
        super(Net_Gaussian, self).__init__()
        self.fc1 = nn.Linear(num_feats, hidden_size, bias= True)
        self.ReLU = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias= True)
        self.ReLU2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, num_classes, bias= True)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ReLU(out)
        out = self.fc2(out)
        out = self.ReLU2(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out
    
    
def train_net(num_epochs, train_load, net, optimizer, criterion, boost=False, sample_weights=None):
    '''
    Trains a model using one optimization criterion for a certain number of epochs.
    Returns training performance indicators over epochs.
    Parameters
    ----------
    num_epochs: int
        Number of training epochs
    train_load: DataLoader
        Training loader
    net: torch.nn.Module
        model to be trained
    optimizer: PyTorch optimizer
        PyTorch optimizer
    criterion: PyTorch criterion
        PyTorch criterion
    boost: bool
        boosting flag
    sample_weights: array(floats)
        boosting weights over samples
        
    Returns
    -------
    loss_epochs: list(FloatTensor)
        training loss over epochs
    '''
    
    net = net.float()
    loss_epochs = []
    for epoch in range(num_epochs):
        loss_epoch = 0
        total = 0
        for i, (feats, labels, idx) in enumerate(train_load):
            feats = feats.view(feats.shape[0], -1)
            feats = Variable(feats.float())
            labels = Variable(labels)
            outputs = net(feats)
            if boost:
                loss = sample_weights[idx] @ criterion(outputs, labels.long())
                loss_epoch += loss
            else:
                loss = criterion(outputs, labels.long())
                loss_epoch = loss / 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epochs.append(loss_epoch)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_epoch.data:.2f}')

    return loss_epochs


def test_net(test_load, net, criterion):
    '''
    Evaluate the model on the test dataset.
    
    Parameters
    ----------
    test_load: DataLoader
        Test loader
    net: torch.nn.Module
        model to be evaluated
    criterion: PyTorch criterion
        PyTorch criterion
        
    Returns
    -------
    acc: FloatTensor
        test accuracy
    _ : FloatTensor
        test loss
    ''' 
    
    net = net.float()
    correct, total = 0, 0
    loss_test = []
    for feats, labels in test_load:
        with torch.no_grad():
            feats = feats.view(1, -1)
            outputs = net(feats.float())
        loss_test.append(criterion(outputs,labels.long()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = torch.true_divide(100 * correct, total)
    return accuracy, sum(loss_test) / len(loss_test)


def compute_all_means(Net, train_size, seed=SEED):
    '''
    Compute the empirical training means for all agents.
    
    Parameters
    ----------
    Net: List(torch.nn.Module)
        List of models
    train_size: int
        number of training samples
    seed: int
        random seed

    Returns
    -------
    _ : array(float)
        array of empirical training means
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_load, _, _, _ = generate_bin_MNIST(batch_size, 
                                             num_agents_sqr, 
                                             [0, 1], 
                                             train_size)
    mean = []
    for i in range(num_agents):
        mean.append(compute_emp_mean(train_load[i], 
                                     Net[i]))
    return np.array(mean)


def compute_emp_mean(train_load, net):
    '''
    Compute the empirical training means for one agent.
    
    Parameters
    ----------
    train_load: DataLoader
        train loader
    net: nn.Module
        model
            
    Returns
    -------
    _ : float
        empirical training mean
    '''
    
    auxsum, total = 0, 0
    for feats, labels, idx in train_load:
        with torch.no_grad():
            feats = feats.view(feats.shape[0], -1)
            outputs = net(feats.float())
        aux = outputs.detach().numpy()
        aux2 = np.log(aux[:,0] / aux[:,1])
        total += len(labels)
        auxsum += np.sum(aux2)
    return auxsum / total


def train_test_agent(mnist_input, train_loader, test_loader, learning_rate, setting=1):
    '''
    Trains and test the model for one agent.
    Returns training and validation performance indicators over epochs.
    Returns the model to be used in the prediction phase.
    
    Parameters
    ----------
    mnist_input: int
        input dimension
    train_loader: DataLoader
        Train loader
    test_loader: DataLoader
        Test loader
    learning_rate: float
        learning rate
    setting: int
        setting selection 
        (1 = first example, 2 = second example, 3 = multiclass example, 4 = Gaussian example)
    
    Returns
    -------
    loss_epochs: list(FloatTensor)
        training loss over epochs
    acc_test: FloatTensor
        test accuracy
    ltest : FloatTensor
        test loss
    net: nn.Module
        trained model
    '''
    
    if setting==1:
        net = Net(mnist_input, 
                  hidden_size, 
                  num_classes)
    elif setting==2:
        net = Net(mnist_input, 
                  hidden_size_simple, 
                  num_classes)
    elif setting==3:
        net = Net(mnist_input, 
                  hidden_size, 
                  num_multiclasses)
    else:
        net = Net_Gaussian(dim_g, 
                           hidden_size_g, 
                           num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), 
                                 lr=learning_rate)
    loss_epochs = train_net(num_epochs, 
                            train_loader, 
                            net, 
                            optimizer, 
                            criterion)
    acc_test, ltest = test_net(test_loader, 
                               net, 
                               criterion)

    return loss_epochs, acc_test, ltest, net


def train_test_agent_boosting(mnist_input, train_loader, test_loader, sample_weights, learning_rate, setting=1):
    '''
    Trains and test the model for one agent for the boosting strategy.
    Returns training loss over epochs.
    Returns the model and boosting score to be used in the prediction phase.
    
    Parameters
    ----------
    mnist_input: int
        input dimension
    train_loader: DataLoader
        Train loader
    test_loader: DataLoader
        Test loader
    sample_weights: FloatTensor
        vector of boosting weights over samples
    learning_rate: float
        learning rate
    setting: int
        setting selection 
        (1 = first example, 2 = second example)
        
    Returns
    -------
    loss_epochs: list(FloatTensor)
        training loss over epochs
    net: nn.Module
        trained model
    class_score: FloatTensor
        vector of boosting weights over classifiers
    sample_weights: FloatTensor
        vector of boosting weights over samples
    '''
    if setting==1:
        net = Net(mnist_input, 
                  hidden_size, 
                  num_classes)
    else:
        net = Net(mnist_input, 
                  hidden_size_simple, 
                  num_classes)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(), 
                                 lr=learning_rate)
    loss_epochs = train_net(num_epochs, 
                            train_loader, 
                            net, 
                            optimizer, 
                            criterion, 
                            True, 
                            sample_weights)
    class_score, sample_weights = acc_boosting(train_loader, 
                                               net, 
                                               sample_weights)

    return loss_epochs, net, class_score, sample_weights


def acc_boosting(train_loader, net, sample_weights):
    '''
    Compute training accuracy of an agent for boosting.
    
    Parameters
    ----------
    train_loader: DataLoader
        Train loader
    net: nn.Module
        model
    sample_weights: FloatTensor
        vector of boosting weights over samples

    Returns
    -------
    class_score: FloatTensor
        vector of boosting weights over classifiers
    sample_weights: FloatTensor
        vector of boosting weights over samples
    '''
    
    net = net.float()
    boosting_error = 0
    wellclass_idx, missclass_idx = [], []

    for feats, labels, idx in train_loader:
        with torch.no_grad():
            feats = feats.view(feats.size(0), -1)
            outputs = net(feats.float())
        _, predicted = torch.max(outputs.data, 1)
        missclass_idx.append(idx[predicted != labels])
        wellclass_idx.append(idx[predicted == labels])

    misclassified_samples = torch.cat(missclass_idx, dim=0)
    wellclassified_samples = torch.cat(wellclass_idx, dim=0)

    boosting_error = sample_weights[misclassified_samples].sum()

    class_score = 0.5 * torch.log((1 - boosting_error) / (boosting_error + 1e-8))

    sample_weights[misclassified_samples] = sample_weights[misclassified_samples] * torch.exp(class_score)
    sample_weights[wellclassified_samples] = sample_weights[wellclassified_samples] * torch.exp(- class_score)
    sample_weights = sample_weights / sample_weights.sum()

    return class_score, sample_weights


def compute_all_means_multiclass(Net):
    '''
    Compute the empirical training means for all agents.
    
    Parameters
    ----------
    Net: List(torch.nn.Module)
        List of models
    train_size: int
        number of training samples
    seed: int
        random seed

    Returns
    -------
    _ : array(float)
        array of empirical training means
    '''
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_load, _, _, _ = generate_multi_MNIST(batch_size, 
                                               num_agents_sqr, 
                                               classes=range(num_multiclasses), 
                                               train_size=train_size)
    mean = []
    for i in range(num_agents):
        mean.append(compute_emp_mean(train_load[i], Net[i]))
        
    return np.array(mean)


def compute_emp_mean_multiclass(train_load, net):
    '''
    Compute the empirical training means for one agent for the multiclass example.
    
    Parameters
    ----------
    train_load: DataLoader
        train loader
    net: nn.Module
        model
            
    Returns
    -------
    _ : array(float)
        empirical training means
    '''
    auxsum, total = np.zeros(num_multiclasses - 1), np.zeros(num_multiclasses - 1)
    for feats, labels, idx in train_load:
        with torch.no_grad():
            feats = feats.view(feats.shape[0], -1)
            outputs = net(feats.float())
        
        aux = outputs.detach().numpy()
        aux2 = np.array([np.log(aux[:,0]/aux[:,i]) for i in range(1, num_multiclasses)])
        for i, l in enumerate(labels):
            if l == 0:
                total += 1
                auxsum += aux2[:, i]
            else:
                total[l-1] += 1
                auxsum[l-1] += aux2[l-1, i]
                
    return auxsum/total


def compute_all_means_Gaussian(Net, train_load):
    '''
    Compute the empirical training means for all agents for the Gaussian example.
    
    Parameters
    ----------
    Net: List(torch.nn.Module)
        List of models
    train_load: DataLoader
        Train loader
        
    Returns
    -------
    _ : array(float)
        array of empirical training means
    '''
    mean = []
    for i in range(num_agents_g):
        mean.append(compute_emp_mean(train_load[i], 
                                     Net[i]))
    return np.array(mean)