import numpy as np
import pickle
def softmax(x):
    s = np.sum(np.exp(x), axis=1).reshape(-1,1)
    return np.exp(x)/s

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def initialize_params(n_x, n_h, n_y):
    wi = np.random.randn(n_h, n_x)
    wf = np.random.randn(n_h, n_x)
    wo = np.random.randn(n_h, n_x)
    wc = np.random.randn(n_h, n_x)
    wy = np.random.randn(n_y, n_h)

    bi = np.zeros((n_h, 1))
    bf = np.zeros((n_h, 1))
    bo = np.zeros((n_h, 1))
    bc = np.zeros((n_h, 1))
    by = np.zeros((n_y, 1))

    params = {"bf": bf, "wf": wf, "bi": bi, "wi": wi,\
               "bc": bc, "wc": wc, "bo": bo, "wo": wo, "wy": wy, "by": by}
    return params


def lstmcell_forward(xt, h_prev, c_prev, params):
    """
    Parameters:
    xt: input at timestep t, shape (n_x, batch_size)
    h_prev: hidden state at timestep t-1, shape (n_h, batch_size)
    c_prev: memory state at timestep t-1, shape (n_h, batch_size)
    params:
        wf: weight matrix of the forget gate, shape (n_h, n_h + n_x)
        bf: bias of the forget gate, shape (n_h, 1)
        wi: weight matrix of the update gate, shape (n_h, n_h + n_x)
        bi: bias of the update gate, shape (n_h, 1)
        wc: weight matrix of tanh activation, shape (n_h, n_h + n_x)
        bc: bias of tanh activation, shape (n_h, 1)
        wo: weight matrix of the output gate, shape (n_h, n_h + n_x)
        bo: bias of the output gate, shape (n_h, 1)
        wy: weight matrix of the output hidden state, shape (n_y, n_h)
        by: bias of the output hidden state, shape (n_y, 1)

    Returns:
    h_next: next hidden state, shape (n_h, batch_size)
    c_next: next memory state, shape (n_h, batch_size)
    yt_pred: prediction at timestep t, shape (n_y, batch_size)
    bprop_vals: tuple of (h_next, c_next, h_prev, c_prev, xt, params) for backpropagation
    """
    # Grab variables
    wf = params["wf"]
    bf = params["bf"]
    wi = params["wi"]
    bi = params["bi"]
    wc = params["wc"]
    bc = params["bc"]
    wo = params["wo"]
    bo = params["bo"]
    wy = params["wy"]
    by = params["by"]

    # Define shapes
    n_x, batch_size = xt.shape
    n_y, n_h = wy.shape

    # Concatenate h_prev, xt
    c_input = np.zeros((n_h + n_x, batch_size))
    c_input[:n_h:] = h_prev
    c_input[n_h:,:]= xt

    ft = sigmoid(np.dot(wf, c_input) + bf) # forget gate
    it = sigmoid(np.dot(wi, c_input) + bi) # update gate
    cct = np.tanh(np.dot(wc, c_input) + bc) # tanh activation
    c_next = (ft * c_prev) + (it * cct) # next memory state
    ot = sigmoid(np.dot(wo, c_input) + bo) # output gate
    h_next = ot * np.tanh(c_next) # next hidden state
    yt_pred = softmax(np.dot(wy, h_next) + by) # prediction of LSTM cell

    bprop_vals = (h_next, c_next, h_prev, c_prev, ft, it, cct, ot, xt, params)
    return h_next, c_next, yt_pred, bprop_vals

def lstmcell_backward(dh_next, dc_next,bprop_vals):
    """
    Parameters:
    dh_next_forward: gradient of next hidden state
    dc_next_forward: gradient of next cell state
    bprop_vals: important values defined in forward pass

    Returns:
    grads:
        dxt: gradient of input at timestep t
        dh_prev: gradient w.r.t. the previous hidden state
        dc_prev: gradient w.r.t. the previous memory state
        dwf: gradient w.r.t. the weight matrix of the forget gate
        dbf: gradient w.r.t. the biases of the forget gate
        dwi: gradient w.r.t. the weight matrix of the update gate
        dbi: gradient w.r.t. the biases of the update gate
        dwc: gradient w.r.t. the weight matrix of the memory gate
        dbc: gradient w.r.t. the biases of the memory gate
        dwo: gradient w.r.t. the weight matrix of the output gate
        dbo: gradient w.r.t. the biases of the output gate
    """
    (h_next, c_next, h_prev, c_prev, ft, it, cct, ot, xt, params) = bprop_vals

    # Define shapes
    n_x, batch_size = xt.shape
    n_h, batch_size = h_next.shape

    # Find gradients for gates
    dot = dh_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot (1 - np.square(np.tanh(c_next))) * it * dh_next) * (1 - np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * dh_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * dh_next) * ft * (1 - ft)



    c_input = np.concatenate((h_prev, xt), axis=0)

    # Find gradients for weights, biases
    dwo = np.dot(dot, c_input.T)
    dbo = np.sum(dot, axis = 1, keepdims = True)
    dwc = np.dot(dcct, c_input.T)
    dbc = np.sum(dcct, axis = 1, keepdims = True)
    dwi = np.dot(dit, c_input.T)
    dbi = np.sum( dit, axis = 1, keepdims = True)
    dwf = np.dot(dft, c_input.T)
    dbf = np.sum(dft, axis = 1, keepdims = True)


    dh_prev = np.dot(params["wf"][:,:n_h].T, dft) + np.dot(params["wi"][:,:n_h].T, dit)\
          + np.dot(params["wc"][:,:n_h].T, dcct) + np.dot(params["wo"][:,:n_h].T, dot)
    
    dc_prev = dh_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * dh_next
    
    dxt = np.dot(params["wf"][:,n_h:].T, dft) + np.dot(params["wi"][:,n_h:].T, dit)\
          + np.dot(params["wc"][:,n_h:].T, dcct) + np.dot(params["wo"][:,n_h:].T, dot)
    
    grads = {"dxt": dxt, "dc_prev": dc_prev, "dh_prev": dh_prev, "dbf": dbf,\
              "dwf": dwf, "dbi": dbi, "dwi": dwi,\
                  "dbc": dbc, "dwc": dwc, "dbo": dbo,\
                      "dwo": dwo}
    return grads


def bilstm_forward(x, h0, params):
    """
    x: inputs for each timestep, shape (n_x, batch_size, t_x)
    h0: initial hidden state, shape (n_h, batch_size)
    params: as defined in lstmcell_forward() 

    Returns:
    h: hidden states for each timestep, shape (n_h, batch_size, t_x)
    y: predictions for each timestep, shape (n_y, batch_size, t_x)
    bilstm_bpropvals: tuple of (list of values per timestep, x)
    """
    lst_vals = []

    n_x, batch_size, t_x = x.shape
    n_y, n_h = params["wy"].shape

    h = np.zeros((n_h, batch_size, t_x))
    c = h
    y = np.zeros((n_y, batch_size, t_x))

    h_forward = np.zeros((n_h, batch_size))
    c_forward = np.zeros((n_h, batch_size))
    h_backward = np.zeros((n_h, batch_size))
    c_backward = np.zeros((n_h, batch_size))

    for t in range(t_x):
        h_forward, c_forward, yt, vals = lstmcell_forward(x[:,:,t], h_forward, c_forward, params)
        h[:n_h, :, t] = h_forward
        y[:, :, t] = yt
        c[:n_h, :, t] = c_forward

        lst_vals.append(vals)

    for t in range(t_x - 1, -1, -1):
        h_backward, c_backward, yt, vals = lstmcell_forward(x[:,:,t], h_backward, c_backward, params)
        h[n_h:,:,t] = h_backward
        c[n_h:,:,t] = c_backward

        lst_vals.append(vals)

    bilstm_bpropvals = (lst_vals, x)
    return h,y,c,bilstm_bpropvals


def bilstm_backward(dh, bilstm_bpropvals):
    """
    
    """
    n_h, batch_size, t_x = dh.shape
    t_x = t_x / 2
    (bprop_vals, x) = bilstm_bpropvals
    forward = bprop_vals[:t_x]
    backward = bprop_vals[t_x:]
    (h1_forward , c1_forward, h0_forward, c0_forward, f1_forward, i1_forward, cct1_forward, o1_forward, x1_forward, params_forward) = forward[0]
    (h1_backward , c1_backward, h0_backward, c0_backward, f1_backward, i1_backward, cct1_backward, o1_backward, x1_backward, params_backward) = backward[0]
    
    n_x, batch_size = x1_forward.shape

    dx = np.zeros((n_x, batch_size, t_x))
    dh0 = np.zeros((n_h, batch_size))
    dh_prev = np.zeros(dh0.shape)
    dc_prev = np.zeros(dh0.shape)
    dwf = np.zeros((n_h, n_h + n_x))
    dwi = np.zeros(dwf.shape)
    dwc = np.zeros(dwf.shape)
    dwo = np.zeros(dwf.shape)
    dbf = np.zeros((n_h,1))
    dbi = np.zeros(dbf.shape)
    dbc = np.zeros(dbf.shape)
    dbo = np.zeros(dbf.shape)

    for t in reversed(range(t_x)):
        grad_forward = lstmcell_backward(dh_prev[:,:,t], dc_prev, forward[t])
        grad_backward = lstmcell_backward(dh_prev[:,:,t], dc_prev, backward[t_x - 1 - t])
        dx[:,:,t] = grad_forward["dxt"] + grad_backward["dxt"]
        dwf += grad_forward["dwf"] + grad_backward["dwf"]
        dwi += grad_forward["dwi"] + grad_backward["dwi"]
        dwc += grad_forward["dwc"] + grad_backward["dwc"]
        dwo += grad_forward["dwo"] + grad_backward["dwo"]
        dbf += grad_forward["dbf"] + grad_backward["dbf"]
        dbi += grad_forward["dbi"] + grad_backward["dbi"]
        dbc += grad_forward["dbc"] + grad_backward["dbc"]
        dbo += grad_forward["dbo"] + grad_backward["dbo"]
    dh0 = grad_forward["dh_prev"] + grad_backward["dh_prev"]

    grads = {"dx": dx, "dh0": dh0, "dbf": dbf, "dwf": dwf, "dbi": dbi, "dwi": dwi,\
              "dbc": dbc, "dwc": dwc, "dbo": dbo, "dwo": dwo}
    return grads

def update_gradients(params, grads, learning_rate=0.001):
    """
    Updates the parameters using the computed gradients.

    Parameters:
    params: dictionary containing the current parameters (weights and biases)
    grads: dictionary containing the gradients (dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, etc.)
    learning_rate: the learning rate for parameter updates
    
    Returns:
    updated_params: dictionary with updated parameters
    """
    updated_params = {}

    # Update weights and biases for each gate in the LSTM cell and the output layer
    updated_params['wf'] = params['wf'] - learning_rate * grads['dwf']
    updated_params['wi'] = params['wi'] - learning_rate * grads['dwi']
    updated_params['wc'] = params['wc'] - learning_rate * grads['dwc']
    updated_params['wo'] = params['wo'] - learning_rate * grads['dwo']
    #updated_params['wy'] = params['wy'] - learning_rate * grads['dwo']  # If needed
    updated_params['bf'] = params['bf'] - learning_rate * grads['dbf']
    updated_params['bi'] = params['bi'] - learning_rate * grads['dbi']
    updated_params['bc'] = params['bc'] - learning_rate * grads['dbc']
    updated_params['bo'] = params['bo'] - learning_rate * grads['dbo']
    #updated_params['by'] = params['by'] - learning_rate * grads['dbo']  # If needed

    return updated_params

# Dense Layer for Classification
class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.zeros((output_dim, 1))

    def forward(self, x):
        return np.dot(self.W, x.T).T + self.b.T

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m