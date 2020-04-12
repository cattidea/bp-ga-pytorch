import numpy as np
import torch

def batch_loader(x, y, batch_size=64):
    data_size = x.shape[0]
    permutation = np.random.permutation(data_size)
    for i in range(0, data_size, batch_size):
        batch_permutation = permutation[i: i+batch_size]
        yield x[batch_permutation], y[batch_permutation]

def train_model(model, x, y, num_epoches=10000, batch_size=4096, learning_rate=1e-3, log=True):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_recorder = []
    for e in range(num_epoches):
        loss_recorder.clear()
        for i, (x_batch, y_batch) in enumerate(batch_loader(x, y, batch_size=batch_size)):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_recorder.append(loss.item())
            if log:
                print('batch {} loss {}'.format(
                    i+1, loss.item()
                ), end='\r')
        if log:
            print('epoch {} loss {}'.format(
                e+1, np.mean(loss_recorder)
            ))
    return np.mean(loss_recorder)

def reset_model(model):
    for layer in model:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

if __name__ == '__main__':

    # Generate Data
    def F(x):
        return 1 / (1 + np.exp(x))

    X_BOUND = [-10, 10]
    data_length = 1000
    noise = 0.1
    x_origin = X_BOUND[0] + (X_BOUND[1]-X_BOUND[0]) * np.random.rand(data_length)
    y_origin = F(x_origin) + noise * np.random.rand(data_length)

    # Build Model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Working on {}'.format(device))
    x_tensor = torch.from_numpy(x_origin).unsqueeze_(-1).float().to(device)
    y_tensor = torch.from_numpy(y_origin).unsqueeze_(-1).float().to(device)

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    ).to(device)

    # Train
    train_model(model, x_tensor, y_tensor, num_epoches=1000, batch_size=4096)
    # Plot
    x_axis = np.linspace(*X_BOUND, 200)
    plt.plot(x_axis, F(x_axis))
    plt.scatter(x_origin, np.squeeze(model(x_tensor).detach().cpu().numpy(), -1), color='r')
    plt.show()
