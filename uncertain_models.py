from parameters import *

def update_hyper_r(k0, v0, omega0, S0, w):
    '''
    Update hyperparameters of the trained model with new sample.
    
    Parameters
    ----------
    k0: int
        hyperparameter 
    v0: int
        hyperparameter
    omega0: ndarray
        prior mean
    S0: ndarray
        prior covariance
    w: ndarray
        data sample
    
    Returns
    -------
    k: int
        updated hyperparameter 
    v: int
        updated hyperparameter
    omega: ndarray
        updated mean
    S: ndarray
        updated covariance
    Shat: ndarray
        normalized covariance
    vhat: float
        normalized hyperparameter
    '''
    H = k0.shape[1]
    d = omega0.shape[2]
    
    k = k0 + 1
    v = v0 + 1
    omega = (k0[:,:,None] * omega0 + np.swapaxes(np.tile(w,(H, 1, 1)), 0, 1))/k[:,:,None]
    S = S0 + np.swapaxes(np.tile(np.einsum('...i,...j->...ij', w, w), (H, 1, 1, 1)), 0, 1) + k0[:, :, None, None] * np.einsum('...i,...j->...ij', omega0, omega0) - k[:, :, None, None] * np.einsum('...i,...j->...ij', omega, omega)
    
    vhat = v - d + 1
    Shat = ((k + 1) / (k * vhat))[:, :, None, None] * S
            
    return k, v, omega, S, Shat, vhat


def update_hyper(k0, v0, omega0, S0, w):
    '''
    Update hyperparameters of the ignorance model with new sample.
    
    Parameters
    ----------
    k0: int
        hyperparameter 
    v0: int
        hyperparameter
    omega0: ndarray
        prior mean
    S0: ndarray
        prior covariance
    w: ndarray
        data sample
    
    Returns
    -------
    k: int
        updated hyperparameter 
    v: int
        updated hyperparameter
    omega: ndarray
        updated mean
    S: ndarray
        updated covariance
    Shat: ndarray
        normalized covariance
    vhat: float
        normalized hyperparameter
    '''
    d = omega0.shape[1]

    k = k0 + 1
    v = v0 + 1
    omega = (k0[:,None] * omega0 + w)/k[:,None]
    S = S0 + np.einsum('...i,...j->...ij', w, w) + k0[:, None, None] * np.einsum('...i,...j->...ij', omega0, omega0) - k[:, None, None] * np.einsum('...i,...j->...ij', omega, omega)
    
    vhat = v - d + 1
    Shat = ((k + 1) / (k * vhat))[:, None, None] * S
                
    return k, v, omega, S, Shat, vhat


def generate_likelihood_r(omega, Shat, vhat, wi):
    '''
    Update normalization factors from ignorance model to be used in the uncertain likelihood update.
    
    Parameters
    ----------
    omega: ndarray
        estimated mean
    Shat: ndarray
        estimated covariance
    vhat: int
        hyperparameter
    wi: ndarray
        data sample
    
    Returns
    -------
    aux: ndarray
        normalization factor    
    '''
    
    N = omega.shape[0]
    H = omega.shape[1]
    
    aux = np.zeros((N, H))
    for l in range(N):
        for h in range(H):
            rv = st.multivariate_t(omega[l][h], 
                                   Shat[l][h], 
                                   vhat[l][h])
            aux[l, h] = rv.pdf(wi[l])
            
    return aux


def generate_likelihood(omega, Shat, vhat, wi):
    '''
    Update normalization factors from trained model to be used in the uncertain likelihood update.
    
    Parameters
    ----------
    omega: ndarray
        estimated mean
    Shat: ndarray
        estimated covariance
    vhat: int
        hyperparameter
    wi: ndarray
        data sample
    
    Returns
    -------
    aux: ndarray
        normalization factor    
    '''
    
    N = len(omega)
    aux = np.zeros(N)
    for l in range(N):
        rv = st.multivariate_t(omega[l], 
                               Shat[l], 
                               vhat[l])
        aux[l] = rv.pdf(wi[l])
        
    return aux


def run_uncertain_algo(params_r, params, w, mu0, A, N_ITER):
    '''
    Compute the beliefs over time.
    
    Parameters
    ----------
    params_r: ndarray
        hyperparameters of the trained model
    params: ndarray
        hyperparameters of the model of complete ignorance
    w: ndarray
        prediction data
    mu0: ndarray
        initial beliefs
    A: ndarray
        combination matrix
    N_ITER: int
        number of prediction samples
        
    Returns
    -------
    MU: ndarray
        evolution of beliefs
    '''
    
    mu = mu0.copy()
    k_r, v_r, omega_r, S_r, Shat_r, vhat_r = params_r
    k, v, omega, S, Shat, vhat = params
    
    MU = [mu]
    for i in range(N_ITER):
        l_num = generate_likelihood_r(omega_r, 
                                      Shat_r, 
                                      vhat_r, 
                                      w[:,i])
        l_den = generate_likelihood(omega, 
                                    Shat, 
                                    vhat, 
                                    w[:,i])
        l = l_num / l_den[:, None]
        mu = np.exp(A.T @ np.log(mu)) * l
        k_r, v_r, omega_r, S_r, Shat_r, vhat_r = update_hyper_r(k_r, 
                                                                v_r, 
                                                                omega_r, 
                                                                S_r, 
                                                                w[:,i])
        k, v, omega, S, Shat, vhat = update_hyper(k, 
                                                  v, 
                                                  omega, 
                                                  S, 
                                                  w[:,i])
        MU.append(mu)
        
    return MU


def compute_train_data_params(r):
    '''
    Estimate moments of the training data.
    
    Parameters
    ----------
    r: ndarray
        training data
    
    Returns
    -------
    rm: ndarray
        mean
    rvar: ndrarray
        variance
    '''
    
    rm = np.mean(r, axis=2)
    rvar = np.sum(np.einsum('...i,...j->...ij', r, r), axis=2)
    return rm, rvar


def initialize_train_data(params_0, r):
    '''
    Initialize hyperparameters of the trained model.
    
    Parameters
    ----------
    params_0: ndarray
        parameter vectors
    r: ndarray
        training data
    
    Returns
    -------
    _: tuple(ndarray)
        updated parameters
    '''
    rm, rvar = compute_train_data_params(r)
    d = r.shape[-1]
    k0, v0, omega0, S0 = params_0
    k_r = k0 + r.shape[2]
    v_r = v0 + r.shape[2]
    omega_r = (k0[:,:,None] * omega0 + r.shape[2] * rm)/ k_r[:, :, None]
    S_r = S0 + rvar + k0[:, :, None, None] * np.einsum('...i,...j->...ij', omega0, omega0) -  k_r[:, :, None, None] * np.einsum('...i,...j->...ij', omega_r, omega_r) 
    vhat_r = v_r - d + 1
    Shat_r = ((k_r + 1) / (k_r * vhat_r))[:, :, None, None] * S_r
    
    return (k_r, v_r, omega_r, S_r, Shat_r, vhat_r)


def initialize_no_data(params_1, d):
    '''
    Initialize hyperparameters of the model of complete ignorance.
    
    Parameters
    ----------
    params_1: ndarray
        parameter vectors
    d: int
        dimension of data
    
    Returns
    -------
    _: tuple(ndarray)
        updated parameters
    '''
    
    k1, v1, omega1, S1 = params_1
    k = k1.copy()
    v = v1.copy()
    omega = omega1.copy()
    S = S1.copy()
    vhat = v - d + 1
    Shat = ((k + 1) / (k * vhat))[:, None, None]* S
    
    return (k, v, omega, S, Shat, vhat)
