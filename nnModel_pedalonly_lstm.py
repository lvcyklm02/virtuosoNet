import torch
import torch.nn as nn

class CustomMaskedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_state_keep_mask):
        super(CustomMaskedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.hidden_state_keep_mask = hidden_state_keep_mask

    def forward(self, x, hidden=None):
        if hidden==None:
            # Get initial hidden state (h_0, c_0) - batch_size, hidden_size
            batch_size = x.size(1)
            h_0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
            c_0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        else:
            print('hidden contained:', len(hidden))
            h_0, c_0 = hidden  # Unpack provided hidden state

        # Output to collect the predictions for each timestep
        output = []

        # Initial hidden state for the first timestep
        h_t = h_0
        c_t = c_0

        # Process each timestep manually
        for t in range(x.size(0)):  # iterate over timesteps
            # Forward the LSTM for the current timestep with the modified h_t
            out, (h_t, c_t) = self.lstm(x[t].unsqueeze(0), (h_t, c_t))

            # Zero out part of the hidden state using mask
            h_t[:, :, torch.logical_not(self.hidden_state_keep_mask.bool())] = 0
            out = h_t

            # Store the output for the current timestep
            output.append(out.squeeze(0))

        # Concatenate the outputs from all timesteps
        output = torch.stack(output, dim=0)
        return output, (h_t, c_t)

# Example usage
input_size = 10
hidden_size = 20
seq_len = 5
batch_size = 3
mask = torch.zeros((hidden_size))
mask[3:4] = 1

# Initialize model
model = CustomMaskedLSTM(input_size, hidden_size, mask)

# Example input tensor (seq_len, batch_size, input_size)
x = torch.randn(seq_len, batch_size, input_size)

# Forward pass through the custom Masked LSTM
output = model(x)
# print(output.shape)  # Should be (seq_len, batch_size, hidden_size)
# print("output",output)
