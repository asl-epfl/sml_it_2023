from nn import *
from social_learning import *
from uncertain_models import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from cycler import cycler
import pickle

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
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.2


def train_test_uncertain_Gaussian():
    '''
    Train uncertain models for the Gaussian example. 
    Save training/prediction performance for plotting.
    '''
    
    # initial beliefs
    mu0 = np.ones((num_agents_g, num_classes_g))

    # initial cond. (ignorance model)
    k0 = np.array([[1] * num_classes_g] * num_agents_g)
    v0 = np.array([[dim_g + 2] * num_classes_g] * num_agents_g)
    omega0 = np.array([[np.zeros(dim_g)] * num_classes_g] * num_agents_g)
    S0 = np.array([[np.eye(dim_g)] * num_classes_g] * num_agents_g)

    # initial cond. 2 (ignorance model)
    k1 = np.array([1]* num_agents_g)
    v1 = np.array([dim_g + 2] * num_agents_g)
    omega1 = np.array([np.zeros(2)] * num_agents_g)
    S1 = np.array([np.eye(2)] * num_agents_g)

    MU = []
    for n in Train_sizes:
        print(f'######### Training uncertain models for train size {n} ########')

        r = Q1.rvs((num_agents_g, 
                    num_classes_g, 
                    n))
        r[1,1,:,:] = Q2.rvs(n)

        params_r = initialize_train_data((k0, v0, omega0, S0), r)
        params = initialize_no_data((k1, v1, omega1, S1), dim_g)

        MU_mc = []
        for n in range(N_MC_g):
            w = Q1.rvs((num_agents_g, N_ITER_g))

            MU_mc.append(run_uncertain_algo(params_r, 
                                            params, 
                                            w, 
                                            mu0, 
                                            A_g, 
                                            N_ITER_g))
        MU.append(MU_mc)

    MU = np.array(MU).mean(1)
    pickle.dump(MU, open(DATA_PATH + "uncertain.pkl", "wb"))

    
def train_Gaussian():
    '''
    Train SML for the Gaussian example. 
    Save trained models.
    '''
    
    for n in Train_sizes:
        n_train = num_classes * n
        
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        train_load, test_load, train_ds, test_ds = generate_Gaussian(batch_size, 
                                                                     n_train, 
                                                                     N_TEST_g)
        torch.save(test_load, DATA_PATH + 'test_agents_gaussian_{}.pkl'.format(n))

        loss_ag, ltest_ag, acc_test_ag = [], [], []
        for l in range(num_agents_g):
            print(f'######### Training SML Agent {l+1} for train size {n} ########')
            _, _, _, net = train_test_agent(dim_g, 
                                            train_load[l],
                                            test_load[l], 
                                            learning_rate_g,
                                            setting=4)
            torch.save(net.state_dict(), MODELS_PATH + 'agent_g_{}_{}.pkl'.format(l, n))
    
    
def test_sml_mc(n_train, n_test, n_mc, A_g):
    '''
    Simulate SML prediction phase using trained models for a given training size.
    
    Parameters
    ----------
    n_train = int
        total number of training samples 
    n_test = int
        total number of prediction samples
    n_mc = int
        number of Monte Carlo repetitions
    A_g = ndarray
        combination matrix
        
    Returns
    -------
    Lambda = List(ndarray)
        beliefs (log-ratios) over time
    '''

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    train_load, _, _, _ = generate_Gaussian(batch_size, 
                                            n_train, 
                                            n_test)
    Lambda = []
    for n in range(n_mc):
        _, test_loader, _, _ = generate_Gaussian(batch_size, 
                                                 n_train, 
                                                 n_test)
        mu_0 = np.random.rand(num_agents_g, num_classes)
        mu_0 = mu_0 / np.sum(mu_0, axis=1)[:, None]
        Net, _ = create_graph_of_nn(num_agents_g, 
                                 num_classes,
                                 setting=4,
                                 n_train=n_train)
        emp_means = compute_all_means_Gaussian(Net, train_load)
        Lambda.append(sl(mu_0, 
                         test_loader, 
                         num_agents_g, 
                         Net, 
                         A_g, 
                         emp_means))
        
    return Lambda


def test_Gaussian():
    '''
    Simulate SML prediction phase using trained models.
    Output beliefs are saved for plotting.
    '''

    la_mean, la_std = [], []
    for n in Train_sizes:
        print(f'######### Running prediction of SML for train size {n} ########')
        n_train = n * num_classes
        Lambda_mc = np.array(test_sml_mc(n_train, 
                                         N_TEST_g, 
                                         N_MC_g, 
                                         A_g))
        la_mean.append(Lambda_mc.mean(0))
        la_std.append(Lambda_mc.std(0))
    pickle.dump(la_mean, open(DATA_PATH + "sml.pkl", "wb"))


def plot_comparison():
    '''
    Plot belief evolution during prediction for SML and the uncertain likelihood strategy for the Gaussian example.
    Save figure in dedicated folder.
    '''
    MU = pickle.load(open(DATA_PATH + "uncertain.pkl", "rb"))

    la_mean = pickle.load(open(DATA_PATH + "sml.pkl", "rb"))

    fig = plt.figure(figsize=(14, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax = fig.add_subplot(gs[0])

    for k in range(len(Train_sizes)):
        ax.plot([np.log(MU[k][i][0][0] / MU[k][i][0][1]) for i in range (N_ITER_g)], 
                label=r'$N= %d$' % (Train_sizes[k]))
        ax.set_xlabel(r'$i$', fontsize=16)
        ax.set_ylabel(r'$\boldsymbol{\lambda}_{1,i}$', fontsize=16)
        ax.set_xlim([1, N_ITER_g])
    ax.set_title(r'Uncertain Likelihoods [12]', fontsize=18)
    ax = fig.add_subplot(gs[1])

    for k in range(len(la_mean)):
        ax.plot(np.arange(len(la_mean[k])), 
                la_mean[k][:,0], 
                label=r'$N = %d$' % (Train_sizes[k]))
        ax.set_xlim([0, len(la_mean[0])])
        ax.set_xlabel(r'$i$', fontsize=16)
        ax.set_ylabel(r'$\boldsymbol{\lambda}_{1,i}$', fontsize=16)
    ax.set_title(r'SML', fontsize=18)
    box = ax.get_position()    
    ax.legend(handlelength=1,
             ncol=5,
             bbox_to_anchor=(0.75 * box.x0, 
                             -0.01 - box.y0, 
                             box.width, 
                             0.1 * box.height),
              borderaxespad=0, 
              bbox_transform=fig.transFigure)
    fig.savefig(FIGS_PATH + 'fig10.pdf', bbox_inches='tight')

    
if __name__ == '__main__':
    train_test_uncertain_Gaussian()
    
    train_Gaussian()
    
    test_Gaussian()

    plot_comparison()