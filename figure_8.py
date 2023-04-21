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
                print(f'######### SML (simple) -- Training Agent {l+1} ########')
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
                print(f'######### SML -- Training Agent {l+1}: Repetition {r} ########')
                loss_epochs, _, ltest, net = train_test_agent(mnist_input[l], 
                                                              train_load[l], 
                                                              test_load[l], 
                                                              learning_rate)
                loss_repeat.append(loss_epochs)
                loss_test_repeat.append(ltest)

            loss_ag.append(list(torch.Tensor(loss_repeat).mean(0)))
            torch.save(net.state_dict(), MODELS_PATH + 'agent_{}.pkl'.format(l))
        torch.save((loss_ag), DATA_PATH + 'train_stats.pkl')


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
        torch.save(test_load, DATA_PATH + 'test_agents_simple.pkl')

        sample_weights = 1/train_ds.targets.size(0) * torch.ones(train_ds.targets.size(0))

        loss_ag = []
        class_weight = []
        for l in range(num_agents):
            print(f'######### Boosting -- Training Agent {l + 1} ########')
            loss_epochs, net, class_score, sample_weights = train_test_agent_boosting(mnist_input[l], 
                                                                                      train_load[l],
                                                                                      test_load[l], 
                                                                                      sample_weights, 
                                                                                      learning_rate_simple, 
                                                                                      setting=2)
            class_weight.append(class_score)
            loss_ag.append(loss_epochs)
            torch.save(net.state_dict(), MODELS_PATH + 'agent_boost_{}_simple.pkl'.format(l))
        torch.save((class_weight, loss_ag), DATA_PATH + 'train_stats_boost_simple.pkl')
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


def mc_sl_boost():
    '''
    Simulate monte carlo repetitions of the prediction phase using SML/ Boosting trained models.
    Output probability of errors are saved for plotting.
    '''
    
    _, A = torch.load(DATA_PATH + 'sl.pkl')
     print(f'######### Prediction: Running {N_MC} Monte Carlo runs ########')

    np.random.seed(SEED_SIMPLE)
    torch.manual_seed(SEED_SIMPLE)
    Net, _ = create_graph_of_nn(num_agents, 
                                num_classes, 
                                setting=2)
    mu_0 = np.ones((num_agents, num_classes))
    mu_0 = mu_0 / np.sum(mu_0, axis=1)[:, None]

    emp_means = nn.compute_all_means(Net, 
                                     train_size_simple, 
                                     SEED_SIMPLE)

    score_sl, score_asl, score_boost = [], [], []
    for i in range(N_MC):
        test_loader, test_set = generate_test_shuffled_2(num_agents_sqr, 
                                                         20, 
                                                         classes=[0, 1])
        d = 0.15
        Lamb = sl(mu_0, 
                  test_loader, 
                  num_agents, 
                  Net, 
                  A, 
                  emp_means)
        Lamb_asl = asl(mu_0, 
                       d, 
                       test_loader, 
                       num_agents, 
                       Net, 
                       A, 
                       emp_means)

        score_asl.append([lamb[0] > 0 for lamb in Lamb_asl[:len(Lamb_asl)//2]] +
                         [lamb[0] < 0 for lamb in Lamb_asl[len(Lamb_asl)//2:]])

        score_sl.append([lamb[0] > 0 for lamb in Lamb[:len(Lamb)//2]] +
                        [lamb[0] < 0 for lamb in Lamb[len(Lamb)//2:]])

        N = create_network_of_nn(num_agents, 
                                 num_classes, 
                                 setting=2)
        class_weights,_ = torch.load(DATA_PATH + 'train_stats_boost_simple.pkl')

        MU = test_boosting(test_loader, 
                           num_agents, 
                           N, 
                           class_weights)
        score_boost.append([mu > 0 for mu in MU[:len(MU)//2]] +
                           [mu < 0 for mu in MU[len(MU)//2:]])
        if (i + 1) % 1000 == 0:
            print('Monte Carlo run: ', i + 1)
            
    torch.save((score_boost, score_asl, score_sl), DATA_PATH + 'boost_asl_sl_mc.pkl')


def plot_mc():
    '''
    Plot the probability of error during prediction for SML - SL, SML - ASL and boosting.
    Save the figures in dedicated folder.
    '''

    boost_mc, asl_mc, sl_mc = torch.load(DATA_PATH + 'boost_asl_sl_mc.pkl')

    sl_c = np.sum(sl_mc, 0)
    asl_c = np.sum(asl_mc, 0)
    boost_c = np.sum(boost_mc, 0)

    f, a = plt.subplots(2, 1, figsize=(5, 6))
    a[0].plot(np.arange(1, len(sl_c)), 
              1 - sl_c[1:] / N_MC, 
              ':.', 
              color='C0', 
              label=r'SML - SL', 
              markersize=6)
    a[0].plot(np.arange(1,len(boost_c)+1), 
              1 - boost_c/ N_MC, 
              ':.', 
              color='C2', 
              label=r'AdaBoost', 
              markersize=6)
    a[0].set_yscale('log')
    a[0].set_ylim(0.001, 2)
    a[0].set_xlim(1, len(sl_c)//2 - 1)
    a[0].set_title(r'Stationary Scenario', fontsize=18)
    a[0].tick_params(axis='both', 
                     which='major', 
                     labelsize=13)
    a[0].set_ylabel(r'Prob. of Error', fontsize=16)
    a[0].set_xlabel(r'$i$', fontsize= 16)
    a[0].legend(fontsize=15, 
                loc='upper center', 
                bbox_to_anchor=(0.5, -.27), 
                ncol=3, 
                handlelength=1)
    
    a[1].plot(np.arange(1,len(asl_c)), 
              1 - asl_c[1:] / N_MC, 
              ':.', 
              color='C1', 
              label=r'SML - ASL', 
              markersize=6)
    a[1].plot(np.arange(1,len(boost_c) + 1), 
              1 - boost_c / N_MC, 
              ':.', 
              color='C2', 
              label=r'AdaBoost', 
              markersize=6)
    a[1].set_yscale('log')
    a[1].set_xlim(1, len(asl_c) - 1)
    a[1].set_title(r'Nonstationary Scenario', fontsize=18)
    a[1].tick_params(axis='both', 
                     which='major', 
                     labelsize=13)
    a[1].set_ylabel(r'Prob. of Error', fontsize=16)
    a[1].set_xlabel(r'$i$', fontsize=16)
    a[1].legend(fontsize=15, 
                loc='upper center', 
                bbox_to_anchor=(0.5, -.27), 
                ncol=3, 
                handlelength=1)

    plt.savefig(FIGS_PATH + 'fig8.pdf', bbox_inches='tight')


if __name__ == '__main__':
    train(simplified=True)
    
    train_boost(simplified=True)
    
    mc_sl_boost()

    plot_mc()

