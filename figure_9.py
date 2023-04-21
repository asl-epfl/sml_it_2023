from nn import *
from social_learning import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import networkx as nx
from cycler import *

color_list_cl = ['b', 'r', 'g', 'y', 'c', 'm']
color_list_df = ['#2274A5', '#5FBB97', '#DA462F', '#FFC847', '#B045A9']
color_list_light = ['#99CDEB', '#A9DBC7', '#EDA097', '#FFDA85', '#DDA7DA']
color_list_blues = ['#144866','#2274A5','#56ABDC','#6FA6C3','#B7D2E1']
color_list_yellows = ['#CC8F00','#F5AB00','#FFBC1F','#FFC847','#FFD470']

mpl.style.use('seaborn-deep')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 16
mpl.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
mpl.rcParams['axes.prop_cycle'] = cycler(color=color_list_df)
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.2


def train_multiclass():
    '''
    Train SML models for the multiclass example. 
    Save models in a dedicated folder.
    Save training performance for plotting.
    '''
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_load, test_load, train_ds, test_ds = generate_multi_MNIST(batch_size, 
                                                                  num_agents_sqr, 
                                                                  range(num_multiclasses), 
                                                                  train_size_multi)
    torch.save(test_load, DATA_PATH + 'test_agents_multi.pkl')

    loss_ag, ltest_ag, acc_test_ag = [], [], []
    for l in range(num_agents):
        loss_repeat, acc_test_repeat, loss_test_repeat = [], [], []
        for r in range(train_repetitions):
            print(f'######### SML Multiclass -- Training Agent {l+1}: Repetition {r+1} ########')
            loss_epochs, _, ltest, net = train_test_agent(mnist_input[l], 
                                                          train_load[l], 
                                                          test_load[l], 
                                                          learning_rate,
                                                          setting=3)
            loss_repeat.append(loss_epochs)
            loss_test_repeat.append(ltest)

        loss_ag.append(list(torch.Tensor(loss_repeat).mean(0)))
        torch.save(net.state_dict(), MODELS_PATH + 'agent_{}_multi.pkl'.format(l))
    torch.save((loss_ag), DATA_PATH + 'train_stats_multiclass.pkl')


def test_multiclass():
    '''
    Simulate prediction phase using trained models.
    Output beliefs are saved for plotting.
    '''
    print(f'######### SML Multiclass -- Prediction ########')
    np.random.seed(SEED)
    test_loader = torch.load(DATA_PATH + 'test_agents_multi.pkl')
    mu_0 = np.random.rand(num_agents, num_multiclasses)
    mu_0 = mu_0 / np.sum(mu_0, axis=1)[:, None]
    d = 0.1
    
    Net, _ = create_graph_of_nn(num_agents, 
                                num_multiclasses,
                                setting=3)
    
    _, A = torch.load(DATA_PATH + 'sl.pkl')
    
    emp_means = compute_all_means_multiclass(Net)
    Lamb = asl_multi(mu_0, d, test_loader, num_agents, Net, A)
    torch.save((Lamb,A), DATA_PATH + 'sl_multiclass.pkl')


def plot_multiclass():
    '''
    Plot training performance and belief evolution during prediction for the multiclass example.
    Save figure in dedicated folder.
    '''
    ##### Fig 9 #####
    Lamb, A = torch.load(DATA_PATH + 'sl_multiclass.pkl')
    Lamb = np.array(Lamb)
    mu = get_mu_from_lambda(Lamb)
    
    R = torch.load(DATA_PATH + 'train_stats_multiclass.pkl')
    
    cmap = mpl.cm.get_cmap('Greys')
    color_vec = [cmap(.1), 
                  cmap(.2), 
                  cmap(.3), 
                  cmap(.4), 
                  cmap(.5), 
                  cmap(.6), 
                  cmap(.7), 
                  cmap(.8)]
    
    cmap2 = mpl.cm.get_cmap('hsv')
    color_vec2 = [cmap2(.0), 
                  cmap2(.1), 
                  cmap2(.15), 
                  cmap2(.23), 
                  cmap2(.43), 
                  cmap2(.5), 
                  cmap2(.6), 
                  cmap2(.7), 
                  cmap2(.8), 
                  cmap2(.9)]
    
    fig = plt.figure(figsize = (15, 3.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.8])
    ax = fig.add_subplot(gs[1])
    
    for i in range(len(mu[0,0])):
        ax.plot(mu[:,0, i], 
                color=color_vec2[i], 
                label=r'$\gamma={}$'.format(i))
    box = ax.get_position()    

    for j in range(num_multiclasses):
        ax.text(15 + j * 100, 
                1.2, 
                'digit {}'.format(j), 
                fontsize=13)
        ax.annotate('', 
                    xy=(0, 1.1), 
                    xycoords='axes fraction', 
                    xytext=((j + 1) * 0.1, 1.1),
        arrowprops=dict(arrowstyle="|-|", mutation_scale=2))

    ax.legend(handlelength=.7, 
              loc='upper left',
              bbox_to_anchor=(box.x1+.03, box.y0, box.width, box.height), 
              borderaxespad=0, 
              bbox_transform=fig.transFigure)
    
    ax.set_xlim(0, N_TEST_MULTI*num_multiclasses)
    ax.set_ylabel(r'$\boldsymbol{\varphi}_{1,i}(\gamma)$', fontsize=16)
    ax.set_xlabel(r'$i$', fontsize=16)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(np.arange(1, num_epochs + 1), R[0], ':o', color='C0', label='agent 1', markersize=4)
    ax1.plot(np.arange(1, num_epochs + 1), R[8], ':o', color='C4', label='agent 9', markersize=4)
    for i in range(1,num_agents-1):
        ax1.plot(np.arange(1, num_epochs + 1), R[i], ':o',color=color_vec[i], markersize=4)
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Average Empirical Risk', fontsize=16)
    ax1.set_xlim([1,num_epochs])
    ax1.set_ylim([1.6, 2.7])
    ax1.legend(loc='upper right',
               fontsize=12,
               handlelength=2)
    plt.savefig(FIGS_PATH + 'fig9.pdf', bbox_inches='tight')
    
if __name__ == '__main__':
    train_multiclass()
    
    test_multiclass()
    
    plot_multiclass()

    