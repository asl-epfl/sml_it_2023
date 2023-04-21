from nn import *
from social_learning import *

import matplotlib.pyplot as plt
import networkx as nx

plt.style.use('seaborn-colorblind')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble":r"\usepackage{bm}"
})

        
def train(simplified=False):
    '''
    Train SML models for the first example and the second example with simplified model. 
    Save models in a dedicated folder.
    Save training performance for plotting.
    
    Parameters
    ----------
    simplified: bool
        simplified model flag
    '''
    
    if simplified:
        np.random.seed(SEED_SIMPLE)
        torch.manual_seed(SEED_SIMPLE)

        train_load, test_load, train_ds, test_ds = generate_bin_MNIST(batch_size, 
                                                                      num_agents_sqr, 
                                                                      [0,1], 
                                                                      train_size_simple)
        torch.save(test_load, DATA_PATH + 'test_agents_simple.pkl')
        loss_ag, ltest_ag, acc_test_ag = [], [], []

        for l in range(num_agents):
            loss_repeat, acc_test_repeat, loss_test_repeat = [], [], []
            for r in range(1):
                print(f'######### SML (simple): Training Agent {l+1}: Repetition {r+1} ########')
                loss_epochs, _, ltest, net = train_test_agent(mnist_input[l], 
                                                              train_load[l], 
                                                              test_load[l], 
                                                              learning_rate_simple, 
                                                              setting=2)
                loss_repeat.append(loss_epochs)
                loss_test_repeat.append(ltest)

            loss_ag.append(list(torch.Tensor(loss_repeat).mean(0)))
            torch.save(net.state_dict(), MODELS_PATH + 'agent_{}_simple.pkl'.format(l))
        torch.save((loss_ag), DATA_PATH + 'train_stats_simple.pkl')
        
    else:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        train_load, test_load, train_ds, test_ds = generate_bin_MNIST(batch_size, 
                                                                      num_agents_sqr, 
                                                                      [0,1], 
                                                                      train_size)
        torch.save(test_load, DATA_PATH + 'test_agents.pkl')

        loss_ag, ltest_ag, acc_test_ag = [], [], []
        for l in range(num_agents):
            loss_repeat, acc_test_repeat, loss_test_repeat = [], [], []
            for r in range(train_repetitions):
                print(f'######### SML: Training Agent {l+1}: Repetition {r+1} ########')
                loss_epochs, _, ltest, net = train_test_agent(mnist_input[l], 
                                                              train_load[l], 
                                                              test_load[l], 
                                                              learning_rate)
                loss_repeat.append(loss_epochs)
                loss_test_repeat.append(ltest)

            loss_ag.append(list(torch.Tensor(loss_repeat).mean(0)))
            torch.save(net.state_dict(), MODELS_PATH + 'agent_{}.pkl'.format(l))
        torch.save((loss_ag), DATA_PATH + 'train_stats.pkl')

        
def test_sl():
    '''
    Simulate prediction phase using trained models.
    Output beliefs are saved for plotting.
    '''
    np.random.seed(SEED)
    _, test_loader, _, _ = generate_bin_MNIST(batch_size, 
                                              num_agents_sqr, 
                                              [0, 1], 
                                              train_size)
    mu_0 = np.random.rand(num_agents, num_classes)
    mu_0 = mu_0 / np.sum(mu_0, axis=1)[:, None]
    d = 0.01
    
    np.random.seed(SEED_GRAPH)
    Net, A = create_graph_of_nn(num_agents, num_classes)
    while not (np.all(np.isclose(np.linalg.matrix_power(A, 1000), np.linalg.matrix_power(A, 1001)))
               and np.all(np.linalg.matrix_power(A, 1000)>0)):
        Net, A = create_graph_of_nn(num_agents, num_classes)
    
    emp_means = compute_all_means(Net, train_size)
    Lamb = asl(mu_0, 
               d, 
               test_loader, 
               num_agents, 
               Net, 
               A, 
               emp_means)
    torch.save((Lamb, A), DATA_PATH + 'sl.pkl')

    
def train_boost(simplified=False):
    '''
    Train boosting models for the first example and the second example with simplified model. 
    Save models in a dedicated folder.
    Save training performance for plotting.
    
    Parameters
    ----------
    simplified: bool
        simplified model flag
    '''
    
    if simplified:
        np.random.seed(SEED_SIMPLE)
        torch.manual_seed(SEED_SIMPLE)

        train_load, test_load, train_ds, test_ds = generate_bin_MNIST(batch_size, 
                                                                      num_agents_sqr, 
                                                                      [0, 1], 
                                                                      train_size_simple)
        torch.save(test_load,  DATA_PATH + 'test_agents_simple.pkl')

        sample_weights = 1/train_ds.targets.size(0) * torch.ones(train_ds.targets.size(0))

        loss_ag = []
        class_weight = []
        for l in range(num_agents):
            print(f'######### Boosting: Training Agent {l + 1} ########')
            loss_epochs, net, class_score, sample_weights = train_test_agent_boosting(mnist_input[l],
                                                                                      train_load[l],
                                                                                      test_load[l], 
                                                                                      sample_weights, 
                                                                                      learning_rate_simple, 
                                                                                      setting=2)
            class_weight.append(class_score)
            loss_ag.append(loss_epochs)
            torch.save(net.state_dict(), MODELS_PATH + 'agent_boost_{}_simple.pkl'.format(l))
        torch.save((class_weight, loss_ag),  DATA_PATH + 'train_stats_boost_simple.pkl')
    else:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        train_load, test_load, train_ds, test_ds = generate_bin_MNIST(batch_size, 
                                                                      num_agents_sqr, 
                                                                      [0, 1], 
                                                                      train_size)
        torch.save(test_load, DATA_PATH + 'test_agents.pkl')

        sample_weights = 1/train_ds.targets.size(0) * torch.ones(train_ds.targets.size(0))

        loss_ag = []
        class_weight = []
        for l in range(num_agents):
            loss_repeat, acc_test_repeat, loss_test_repeat = [], [], []
            print(f'######### Boosting (simple): Training Agent {l + 1} ########')
            loss_epochs, net, class_score, sample_weights = train_test_agent_boosting(mnist_input[l], 
                                                                                      train_load[l],
                                                                                      test_load[l], 
                                                                                      sample_weights, 
                                                                                      learning_rate)
            class_weight.append(class_score)
            loss_ag.append(loss_epochs)
            torch.save(net.state_dict(), MODELS_PATH + 'agent_boost_{}.pkl'.format(l))
        torch.save((class_weight,loss_ag), DATA_PATH + 'train_stats_boost.pkl')

        
def test_boost():
    '''
    Simulate prediction phase using boosting trained models.
    Output beliefs are saved for plotting.
    '''
    np.random.seed(SEED)
    _, test_loader, _, _ = generate_bin_MNIST(batch_size, 
                                              num_agents_sqr, 
                                              [0, 1], 
                                              train_size)
    class_weight, loss_ag = torch.load(DATA_PATH + 'train_stats_boost.pkl')
    Net = create_network_of_nn(num_agents, num_classes)
    L = test_boosting(test_loader, 
                      num_agents, 
                      Net, 
                      class_weight)
    torch.save(L, DATA_PATH + 'boostn.pkl')
    

def plot_sl():
    '''
    Plot network topology, training performance and belief evolution during prediction.
    Save figures in dedicated folder.
    '''
    ##### Fig 3 #####
    trainset = datasets.MNIST('mnist_test', 
                              download=True, 
                              train=True, 
                              transform=transforms.ToTensor())
    img = trainset.data[trainset.targets == 0][0]
    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.axis('off')
    ax.imshow(img, cmap = 'Greys')
    ax.plot(10*np.ones(28), 
            color = 'C2', 
            linewidth = 2, 
            linestyle = 'dashed', 
            alpha=.5)
    ax.plot(19*np.ones(28), 
            color = 'C2', 
            linewidth = 2, 
            linestyle = 'dashed', alpha=.5)
    ax.plot(10*np.ones(28), 
            np.arange(28), 
            color = 'C2', 
            linewidth = 2, 
            linestyle = 'dashed', 
            alpha=.5)
    ax.plot(19*np.ones(28), 
            np.arange(28), 
            color = 'C2', 
            linewidth = 2, 
            linestyle = 'dashed', 
            alpha=.5)
    ax.text(5 , 5 , f' 1', 
            fontsize=19,
            color='C0', 
            ha='right', 
            bbox=dict(boxstyle='circle', 
                      facecolor='white', 
                      alpha=.95, linewidth=1.5, 
                      ec='C0'))
    ax.text(25 , 25, f'9', 
            fontsize=19,
            color='C5', 
            ha='right', 
            bbox=dict(boxstyle='circle', 
                      facecolor='white', 
                      alpha=.95, 
                      linewidth=1.5, 
                      ec='C5'))
    
    for i in range(1,8):
        ax.text(5* (2 * (i%num_agents_sqr)+1),
                5 * (2*(i//num_agents_sqr)+1), 
                f' {i+1}', fontsize = 19, 
                color = 'dimgray', 
                ha = 'right',
                bbox=dict(boxstyle = 'circle', 
                          facecolor='white', 
                          alpha=.95,
                          linewidth=1.5, 
                          ec = 'dimgray'))
    ax.set_xlim([0, 28])
    f.savefig(FIGS_PATH + 'fig3.pdf', bbox_inches='tight')

    ##### Fig 4 #####
    Lamb, A = torch.load(DATA_PATH + 'sl.pkl')
    Gr = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    pos = {i: [0.5*(i % num_agents_sqr), - 0.5*(i//num_agents_sqr)] for i in range(num_agents)}
    colors = plt.cm.gray(np.linspace(0.3, 0.8, num_agents))

    f, ax = plt.subplots(1, 1, figsize=(3, 2.7))
    plt.axis('off')
    nx.draw_networkx_nodes(Gr, 
                           pos=pos, 
                           node_color='C0', 
                           nodelist=[0], 
                           node_size=300, 
                           edgecolors='k',
                           linewidths=.5)
    nx.draw_networkx_nodes(Gr, 
                           pos=pos, 
                           node_color='C5', 
                           nodelist=[8], 
                           node_size=300, 
                           edgecolors='k',
                           linewidths=.5)
    nx.draw_networkx_nodes(Gr, 
                           pos=pos, 
                           node_color=colors[8], 
                           nodelist=range(1, num_agents-1), 
                           node_size=300, 
                           edgecolors='k',
                           linewidths=.5)
    nx.draw_networkx_labels(Gr, 
                            pos, 
                            {i: i + 1 for i in range(num_agents)}, 
                            font_size=14, 
                            font_color='black', 
                            alpha=1)
    nx.draw_networkx_edges(Gr, 
                           style='dashed', 
                           pos=pos, 
                           node_size=300, 
                           alpha=1, 
                           arrowsize=10, 
                           width=1, 
                           connectionstyle='arc3,rad=0.1')
    ax.set_ylim(-1.07,0.07)
    f.tight_layout()
    plt.savefig(FIGS_PATH + 'fig4.pdf', bbox_inches='tight')

    ##### Fig 5 #####
    e, v = np.linalg.eig(A)
    pv = np.real(v[:,np.where(np.isclose(e, 1))[0]])[:,0]
    pv = pv/np.sum(pv)

    loss_ag =torch.load(DATA_PATH + 'train_stats.pkl')
    plt.figure(figsize=(5, 2.7))
    plt.plot(np.arange(1, num_epochs + 1), 
             loss_ag[0], 
             ':o', 
             color='C0', 
             label='Agent 1', 
             markersize=4)
    plt.plot(np.arange(1, num_epochs + 1), 
             loss_ag[8], 
             ':o', 
             color='C5', 
             label='Agent 9', 
             markersize=4)
    for i in range(1,num_agents-1):
        plt.plot(np.arange(1, num_epochs + 1), 
                 loss_ag[i], 
                 ':o',
                 color=colors[i], 
                 markersize=4)    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Average Empirical Risk', fontsize=14)
    plt.xlim([1,num_epochs])
    plt.tick_params(axis='both', 
                    which='major', 
                    labelsize=13)
    plt.legend(fontsize=14, handlelength=2)
    plt.savefig(FIGS_PATH + 'fig5.pdf', bbox_inches='tight')

    ##### Fig 6 #####
    plt.figure(figsize=(5, 2.7))
    plt.plot([Lamb[i][0] for i in range(len(Lamb))], 
             linewidth=1.5)
    plt.plot(np.zeros(len(Lamb)), 
             linewidth=1.5, 
             linestyle='dashed', 
             color='C2', 
             alpha=0.9)
    plt.ylabel(r'$\bm{\lambda}_{1,i}$', 
               fontsize=16, 
               labelpad=-10)
    plt.xlabel(r'$i$', 
               fontsize=16,
               labelpad=2)
    plt.xlim([0, len(Lamb)])
    plt.tick_params(axis='both', 
                    which='major', 
                    labelsize=13)

    plt.xticks(np.arange(0, len(Lamb), 1000))
    plt.text(4250, -270, 
             'Digit 0', 
             fontsize=12, 
             bbox=dict(facecolor='white', 
                       linewidth=1.5, 
                       ec='C2'))
    plt.text(4250, 200, 
             'Digit 1', 
             fontsize=12, 
             bbox=dict(facecolor='white', 
                       linewidth=1.5, 
                       ec='C2'))
    plt.tight_layout()
    plt.savefig(FIGS_PATH + 'fig6.pdf', bbox_inches='tight')

    
def plot_boost():
    '''
    Plot the belief evolution during prediction for SML and boosting.
    Save the figures in dedicated folder.
    '''
    ##### Fig 7 #####

    MU = torch.load(DATA_PATH + 'boostn.pkl')
    Lamb, _ = torch.load(DATA_PATH + 'sl.pkl')

    lamb_sl = np.array([np.sign(Lamb[i][0]) for i in range(len(Lamb))])
    lamb_bo = np.array(MU)
    f, a = plt.subplots(2, 1, figsize=(5, 4))
    a[0].scatter(np.arange(len(Lamb))[lamb_sl > 0], 
                 1 * np.ones(sum(lamb_sl > 0)),
                 linewidth=1.5, 
                 marker='.', 
                 s=50,
                 color='C0')
    a[0].scatter(np.arange(len(Lamb))[lamb_sl < 0], 
                 0 * np.ones(sum(lamb_sl < 0)),
                 linewidth=1.5, 
                 marker='.', 
                 s=50,
                 color='C2')
    a[0].set_xlim(0, len(MU))
    a[0].set_ylim(-0.2, 1.2)
    a[0].set_ylabel(r'$\bm{\gamma}^{\sf SML}_{1,i}$', fontsize=16)
    a[0].set_xlabel(r'$i$', fontsize=16)
    a[0].tick_params(axis='both', 
                     which='major', 
                     labelsize=13)
    a[0].set_yticks([0, 1])

    a[1].scatter(np.arange(len(MU))[lamb_bo < 0], 
                 1 * np.ones(sum(lamb_bo < 0)),
                 linewidth=1.5, 
                 marker='.', 
                 s=50, 
                 color='C0')
    a[1].scatter(np.arange(len(MU))[lamb_bo > 0], 
                 0 * np.ones(sum(lamb_bo > 0)),
                 linewidth=1.5, 
                 marker='.', 
                 s=50, 
                 color='C2')
    a[1].set_xlim(0, len(MU))
    a[1].set_ylim(-0.2, 1.2)
    a[1].set_ylabel(r'$\bm{\gamma}^{\sf Boost}_{i}$', fontsize = 16)
    a[1].set_xlabel(r'$i$', fontsize=16)
    a[1].tick_params(axis='both', 
                     which='major', 
                     labelsize=13)
    a[1].set_yticks([0, 1])
    f.savefig(FIGS_PATH + 'fig7.pdf', bbox_inches='tight')


if __name__ == '__main__':
    train()

    test_sl()
    
    plot_sl()
    
    train_boost()
    
    test_boost()
    
    plot_boost()