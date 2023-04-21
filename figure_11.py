from parameters import *
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn-muted')
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{mathrsfs} '
plt.rcParams.update({'text.usetex': True})
mpl.rcParams['font.family'] = 'serif'

def exact_f(x):
    '''
    Compute true function value.
    
    Parameters
    ----------
    x: float
        function argument
    
    Returns
    -------
    y: float
        function output
    '''
    w = 9 * np.exp(2 * x) + (3 * np.exp(3*x) * (-4 + 27 * np.exp(x)))**(1/2)
    z = (2 * 3**(1/3) + 2**(1/3) * np.exp(-x) * w**(2/3)) / (6**(2/3) * w**(1/3))
    y = np.log(z)/4
    return y

def approx_f(x):
    '''
    Compute approximate function value.
    
    Parameters
    ----------
    x: float
        function argument
    
    Returns
    -------
    y: float
        function output
    '''
    
    y = exact_f(0) * (1 - x / np.log(2))
    return y

if __name__ == '__main__':
    x = np.linspace(0, np.log(2), 100)
    y_exact = exact_f(x)
    y_approx = approx_f(x)
    
    # Plot figure
    f, a = plt.subplots(1, 1, figsize=(5, 2.5))
    a.plot(x, y_exact, 
           'k', 
           linewidth=2)
    a.plot(x, y_approx, 
           'C2', 
           marker='o', 
           linestyle=':', 
           linewidth=2, 
           markevery=5, 
           markersize=5)
    a.set_xlabel(r'${\sf R}^o$', fontsize=14)
    a.set_ylabel(r'$\mathscr{E}({\sf R}^o)$', fontsize=14)
    a.set_xlim([0,np.log(2)])
    a.set_ylim([0, 0.1])
    a.tick_params(which='major', labelsize=12)
    a.set_yticks([0, 0.05, 0.10])
    a.legend(['Exact expression -- Eq.~(122)', 'Approximation -- Eq.~(124)'], fontsize=13)
    f.savefig(FIGS_PATH + 'fig11.pdf', bbox_inches='tight')
