import numpy as np

def softmax(x):
    s = np.sum(np.exp(x), axis=1).reshape(-1,1)
    return np.exp(x)/s

def sigmoid(x):
    return 1/(1 + np.exp(-x))

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

def lstmcell_backward(dh_next_forward, dc_next_forward, dh_next_backward, dc_next_backward, bprop_vals):
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
    dot_forward = dh_next_forward * np.tanh(c_next) * ot * (1 - ot)
    dcct_forward = (dc_next_forward * it + ot (1 - np.square(np.tanh(c_next))) * it * dh_next_forward) * (1 - np.square(cct))
    dit_forward = (dc_next_forward * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * dh_next_forward) * it * (1 - it)
    dft_forward = (dc_next_forward * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * dh_next_forward) * ft * (1 - ft)

    dot_backward = dh_next_backward * np.tanh(c_next) * ot * (1 - ot)
    dcct_backward = (dc_next_backward * it + ot (1 - np.square(np.tanh(c_next))) * it * dh_next_backward) * (1 - np.square(cct))
    dit_backward = (dc_next_backward * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * dh_next_backward) * it * (1 - it)
    dft_backward = (dc_next_backward * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * dh_next_backward) * ft * (1 - ft)

    c_input = np.concatenate((h_prev, xt), axis=0)

    # Find gradients for weights, biases
    dwo_forward = np.dot(dot_forward, c_input.T)
    dbo_forward = np.sum(dot_forward, axis = 1, keepdims = True)
    dwc_forward = np.dot(dcct_forward, c_input.T)
    dbc_forward = np.sum(dcct_forward, axis = 1, keepdims = True)
    dwi_forward = np.dot(dit_forward, c_input.T)
    dbi_forward = np.sum( dit_forward, axis = 1, keepdims = True)
    dwf_forward = np.dot(dft_forward, c_input.T)
    dbf_forward = np.sum(dft_forward, axis = 1, keepdims = True)

    dwo_backward = np.dot(dot_backward, c_input.T)
    dbo_backward = np.sum(dot_backward, axis = 1, keepdims = True)
    dwc_backward = np.dot(dcct_backward, c_input.T)
    dbc_backward = np.sum(dcct_backward, axis = 1, keepdims = True)
    dwi_backward = np.dot(dit_backward, c_input.T)
    dbi_backward = np.sum( dit_backward, axis = 1, keepdims = True)
    dwf_backward = np.dot(dft_backward, c_input.T)
    dbf_backward = np.sum(dft_backward, axis = 1, keepdims = True)

    dh_prev = np.dot(params["wf"][:,:n_h].T, dft_forward) + np.dot(params["wi"][:,:n_h].T, dit_forward)\
          + np.dot(params["wc"][:,:n_h].T, dcct_forward) + np.dot(params["wo"][:,:n_h].T, dot_forward)\
              + np.dot(params["wf"][:,:n_h].T, dft_backward) + np.dot(params["wi"][:,:n_h].T, dit_backward)\
                  + np.dot(params["wc"][:,:n_h].T, dcct_backward) + np.dot(params["wo"][:,:n_h].T, dot_backward)
    
    dc_prev = dh_next_forward * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * dh_next_forward\
          + dh_next_backward * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * dh_next_backward
    
    dxt = np.dot(params["wf"][:,n_h:].T, dft_forward) + np.dot(params["wi"][:,n_h:].T, dit_forward)\
          + np.dot(params["wc"][:,n_h:].T, dcct_forward) + np.dot(params["wo"][:,n_h:].T, dot_forward)\
              + np.dot(params["wf"][:,n_h:].T, dft_backward) + np.dot(params["wi"][:,n_h:].T, dit_backward)\
                  + np.dot(params["wc"][:,n_h:].T, dcct_backward) + np.dot(params["wo"][:,n_h:].T, dot_backward)
    
    grads = {"dxt": dxt, "dc_prev": dc_prev, "dh_prev": dh_prev, "dbf_forward": dbf_forward,\
              "dwf_forward": dwf_forward, "dbi_forward": dbi_forward, "dwi_forward": dwi_forward,\
                  "dbc_forward": dbc_forward, "dwc_forward": dwc_forward, "dbo_forward": dbo_forward,\
                      "dwo_forward": dwo_forward,
                  "dbf_backward": dbf_backward, "dwf_backward": dwf_backward, "dbi_backward": dbi_backward,\
                      "dwi_backward": dwi_backward, "dbc_backward": dbc_backward, "dwc_backward": dwc_backward,\
                          "dbo_backward": dbo_backward, "dwo_backward": dwo_backward}
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

    for t in range(t_x):
        grad = lstmcell_backward(dh_prev, dc_prev, dh_prev, dc_prev, )

    
