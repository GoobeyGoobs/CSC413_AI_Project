import numpy as np
import pickle

# LSTM Cell Implementation
class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights for gates
        self.W_i = np.random.randn(hidden_dim, input_dim)
        self.U_i = np.random.randn(hidden_dim, hidden_dim)
        self.b_i = np.zeros((hidden_dim, 1))

        self.W_f = np.random.randn(hidden_dim, input_dim)
        self.U_f = np.random.randn(hidden_dim, hidden_dim)
        self.b_f = np.zeros((hidden_dim, 1))

        self.W_o = np.random.randn(hidden_dim, input_dim)
        self.U_o = np.random.randn(hidden_dim, hidden_dim)
        self.b_o = np.zeros((hidden_dim, 1))

        self.W_c = np.random.randn(hidden_dim, input_dim)
        self.U_c = np.random.randn(hidden_dim, hidden_dim)
        self.b_c = np.zeros((hidden_dim, 1))

    def forward(self, x_t, h_prev, c_prev):
        # Input gate
        i_t = self.sigmoid(np.dot(self.W_i, x_t) + np.dot(self.U_i, h_prev) + self.b_i)
        
        # Forget gate
        f_t = self.sigmoid(np.dot(self.W_f, x_t) + np.dot(self.U_f, h_prev) + self.b_f)
        
        # Output gate
        o_t = self.sigmoid(np.dot(self.W_o, x_t) + np.dot(self.U_o, h_prev) + self.b_o)
        
        # Candidate cell state
        c_tilde_t = np.tanh(np.dot(self.W_c, x_t) + np.dot(self.U_c, h_prev) + self.b_c)
        
        # Current cell state
        c_t = f_t * c_prev + i_t * c_tilde_t
        
        # Current hidden state
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# BiLSTM Implementation
class BiLSTM:
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        
        # Forward and backward LSTM cells
        self.forward_lstm = LSTMCell(input_dim, hidden_dim)
        self.backward_lstm = LSTMCell(input_dim, hidden_dim)
    
    def forward(self, X):
        T, input_dim = X.shape
        h_forward = np.zeros((T, self.hidden_dim))
        h_backward = np.zeros((T, self.hidden_dim))

        # Initialize hidden and cell states
        h_prev_forward = np.zeros((self.hidden_dim, 1))
        c_prev_forward = np.zeros((self.hidden_dim, 1))
        
        h_prev_backward = np.zeros((self.hidden_dim, 1))
        c_prev_backward = np.zeros((self.hidden_dim, 1))

        # Forward pass
        for t in range(T):
            h_prev_forward, c_prev_forward = self.forward_lstm.forward(
                X[t:t+1].T, h_prev_forward, c_prev_forward
            )
            h_forward[t] = h_prev_forward.T

        # Backward pass
        for t in reversed(range(T)):
            h_prev_backward, c_prev_backward = self.backward_lstm.forward(
                X[t:t+1].T, h_prev_backward, c_prev_backward
            )
            h_backward[t] = h_prev_backward.T

        # Concatenate forward and backward outputs
        h_concat = np.concatenate((h_forward, h_backward), axis=1)
        return h_concat

# Dense Layer for Classification
class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.zeros((output_dim, 1))

    def forward(self, x):
        return np.dot(self.W, x.T).T + self.b.T

# Softmax Function
def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Cross-Entropy Loss
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m

# Training Loop
def train_bilstm(X_train, y_train, hidden_dim, output_dim, epochs=10, lr=0.01):
    # Initialize BiLSTM and Dense Layer
    input_dim = X_train.shape[2]
    bilstm = BiLSTM(input_dim, hidden_dim)
    dense = DenseLayer(2 * hidden_dim, output_dim)

    for epoch in range(epochs):
        total_loss = 0
        for i, (x, y) in enumerate(zip(X_train, y_train)):
            # Forward pass through BiLSTM
            h_concat = bilstm.forward(x)
            
            # Forward pass through Dense Layer
            logits = dense.forward(h_concat[-1])  # Use last timestep for classification
            
            # Apply softmax to logits
            y_pred = softmax(logits.reshape(1, -1))
            
            # Compute loss
            loss = cross_entropy_loss(np.array([y]), y_pred)
            total_loss += loss

            # (Optional) Backpropagation logic goes here

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_train):.4f}")

# Load MFCC features
with open("mfcc_features.pkl", "rb") as f:
    X_audio, y_audio = pickle.load(f)

# Reshape MFCCs for BiLSTM
num_samples, input_dim = X_audio.shape
sequence_length = 13  # Define sequence length
X_audio_reshaped = X_audio.reshape(num_samples, sequence_length, input_dim // sequence_length)

# Train the BiLSTM
hidden_dim = 32
output_dim = len(set(y_audio))  # Number of emotion classes
train_bilstm(X_audio_reshaped, y_audio, hidden_dim, output_dim, epochs=5, lr=0.01)


