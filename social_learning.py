from parameters import *
from generate_data import *
import nn
import torch


def generate_sc_graph(num_agents):
    '''
    Generate strongly connected graph.
    
    Parameters
    ----------
    num_agents: int
        number of agents
    Returns
    -------
    G: ndarray
        graph adjacency matrix
    '''
    
    G = np.random.choice([0.0, 1.0], 
                         size=(num_agents, num_agents), 
                         p=[0.8, 0.2])
    G = G + np.eye(num_agents)
    G = (G > 0) * 1.0
    return G


def create_uniform_combination_matrix(G):
    '''
    Generate combination matrix using the uniform rule.
    
    Parameters
    ----------
    G: ndarray
        adjacency matrix
        
    Returns
    -------
    A: ndarray
        combination matrix
    '''
    
    A = G / np.sum(G, 0)
    return A


def create_graph_of_nn(num_agents, num_classes, setting=1, n_train=0):
    '''
    Attribute the training models for each gent.
    
    Parameters
    ----------
    num_agents: int
        number of agents
    num_classes: int
        output size for the NN
    mnist_input: int
        input size for the NN
    setting: int
        setting selection 
        (1 = first example, 2 = second example, 3 = multiclass example, 4 = Gaussian example)

    Returns
    -------
    N: list(nn.Module)
        list of all modules
    A: ndarray
        combination matrix
    '''

    G = generate_sc_graph(num_agents)
    A = create_uniform_combination_matrix(G)
    
    if setting==1:
        N = []
        for i in range(num_agents):
            N.append(nn.Net(mnist_input[i], 
                            hidden_size, 
                            num_classes))
            N[i].load_state_dict(torch.load('models/agent_{}.pkl'.format(i)))
            N[i].eval()
    elif setting==2:
        N = []
        for i in range(num_agents):
            N.append(nn.Net(mnist_input[i], 
                              hidden_size_simple, 
                              num_classes))
            N[i].load_state_dict(torch.load('models/agent_{}_simple.pkl'.format(i)))
            N[i].eval()
    elif setting==3:
        N = []
        for i in range(num_agents):
            N.append(nn.Net(mnist_input[i], 
                            hidden_size, 
                            num_classes))
            N[i].load_state_dict(torch.load('models/agent_{}_multi.pkl'.format(i)))
            N[i].eval()
    else:
        N = []
        for i in range(num_agents_g):
            N.append(nn.Net_Gaussian(dim_g, hidden_size_g, num_classes))
            N[i].load_state_dict(torch.load('models/agent_g_{}_{}.pkl'.format(i, n_train//num_classes)))
            N[i].eval()
       
    return N, A


def asl(mu_0, d, test_loader, num_agents, N, A, emp_mean=0):
    '''
    Run prediction phase using the ASL algorithm.
    
    Parameters
    ----------
    mu_0: ndarray
        initial beliefs
    d: float
        step-size parameter
    test_loader: Dataloader
        test Dataloader
    num_agents: int
        number of agents
    N: list(nn.Module)
        list of NN models
    A: ndarray
        combination matrix
    emp_mean: array(float)
        empirical training mean vector
    
    Returns
    -------
    Lamb: list(ndarray)
        Belief (log-ratio) evolution over time
    '''

    lamb = np.log(mu_0[:,0]/mu_0[:,1])
    Lamb = [lamb]

    for i in range(len(test_loader[0])):
        L=[]
        for j in range(num_agents):
            with torch.no_grad():
                feat = (test_loader[j].dataset.__getitem__(i)[0]).float()
                feat = feat.view(1, -1)
                outputs = N[j](feat)
            L.append(outputs.detach().numpy())
        L = np.array(L)[:,0]
        logL = np.log(L[:,0] / L[:,1]) - emp_mean
        psi_l = (1-d) * lamb +logL
        lamb = A.T @ psi_l
        Lamb.append(lamb)
    return Lamb


def sl(mu_0, test_loader, num_agents, N, A, emp_mean=0):
    '''
    Run prediction phase using the traditional SL algorithm.
    
    Parameters
    ----------
    mu_0: ndarray
        initial beliefs
    test_loader: Dataloader
        test Dataloader
    num_agents: int
        number of agents
    N: list(nn.Module)
        list of NN models
    A: ndarray
        combination matrix
    emp_mean: array(float)
        empirical training mean vector
    
    Returns
    -------
    Lamb: list(ndarray)
        Belief (log-ratio) evolution over time
    '''
    
    lamb = np.log(mu_0[:,0] / mu_0[:,1])
    Lamb = [lamb]

    for i in range(len(test_loader[0])):
        L = []
        for j in range(num_agents):
            with torch.no_grad():
                feat = (test_loader[j].dataset.__getitem__(i)[0]).float()
                feat = feat.view(1, -1)
                outputs = N[j](feat)
            L.append(outputs.detach().numpy())
        L = np.array(L)[:,0]
        logL = np.log(L[:,0] / L[:,1]) - emp_mean
        psi_l = lamb + logL
        lamb = A.T @ psi_l
        Lamb.append(lamb)
    return Lamb


def create_network_of_nn(num_agents, num_classes, setting=1):
    '''
    Attribute the boosting trained models for each gent.
    
    Parameters
    ----------
    num_agents: int
        number of agents
    num_classes: int
        output size for the NN
    mnist_input: int
        input size for the NN
    setting: int
        setting selection 
        (1 = first example, 2 = second example)


    Returns
    -------
    N: list(nn.Module)
        list of all modules
    '''

    if setting==1:
        N = []
        for i in range(num_agents):
            N.append(nn.Net(mnist_input[i], hidden_size, num_classes))
            N[i].load_state_dict(torch.load('models/agent_boost_{}.pkl'.format(i)))
            N[i].eval()
    else:
        N = []
        for i in range(num_agents):
            N.append(nn.Net(mnist_input[i], hidden_size_simple, num_classes))
            N[i].load_state_dict(torch.load('models/agent_boost_{}_simple.pkl'.format(i)))
            N[i].eval()
    return N


def test_boosting(test_loader, num_agents, N, class_weights):
    '''
    Run prediction phase using the boosting strategy.
    
    Parameters
    ----------
    test_loader: Dataloader
        test Dataloader
    num_agents: int
        number of agents
    N: list(nn.Module)
        list of NN models
    class_weights: FloatTensor
        vector of boosting weights over classifiers
    
    Returns
    -------
    Lamb: list(ndarray)
        Boosting classification collected over time
    '''
    Lamb = []
    for i in range(len(test_loader[0])):
        L = []
        for j in range(num_agents):
            with torch.no_grad():
                feat = (test_loader[j].dataset.__getitem__(i)[0]).float()
                feat = feat.view(1, -1)
                outputs = N[j](feat)
            L.append(outputs)
        L = torch.cat(L, dim = 0)

        L = torch.sign(torch.log(L[:,0] / L[:,1]))
        mu = torch.tensor(class_weights) @ L
        Lamb.append(torch.sign(mu))
    return Lamb


def asl_multi(mu_0, d, test_loader, num_agents, N, A, emp_mean=0):
    '''
    Run prediction phase using the ASL algorithm for the multiclass example.
    
    Parameters
    ----------
    mu_0: ndarray
        initial beliefs
    d: float
        step-size parameter
    test_loader: Dataloader
        test Dataloader
    num_agents: int
        number of agents
    N: list(nn.Module)
        list of NN models
    A: ndarray
        combination matrix
    emp_mean: array(float)
        empirical training mean vector
    
    Returns
    -------
    Lamb: list(ndarray)
        Belief (log-ratio) evolution over time
    '''
    
    lamb = np.array([np.log(mu_0[:,0]/mu_0[:,i]) for i in range(1, mu_0.shape[1])])
    Lamb = [lamb]

    for i in range(len(test_loader[0])):
        L=[]
        for j in range(num_agents):
            with torch.no_grad():
                feat = (test_loader[j].dataset.__getitem__(i)[0]).float()
                feat = feat.view(1, -1)
                outputs = N[j](feat)
            L.append(outputs.detach().numpy())
        L = np.array(L)[:,0]
        logL = np.array([np.log(L[:,0]/L[:,i]) for i in range(1, mu_0.shape[1])]).T
        psi_l = (1-d) * lamb + (logL - emp_mean)
        lamb = A.T @ psi_l
        Lamb.append(lamb)
    return Lamb


def get_mu_from_lambda(Lamb):
    '''
    Extract beliefs from log-ratio of beliefs
    
    Parameters
    ----------
    Lamb: list(array(float))
        list of log-ratio of beliefs
    
    Returns
    -------
    _: array(float)
        vector of beliefs
    '''
    Lamb = np.array(Lamb)
    aux = 1 / np.exp(Lamb)
    p1 = 1 / (1 + np.sum(aux, -1))
    p = aux* p1[:, :, None]
    return np.concatenate([p1[:,:, None], p], 2)