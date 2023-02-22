import torch
import argparse
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import utils
from main_logger import get_logger
from gradient_vector import *
from kernels import *
from query_functions import *
from blackbox_models import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help="Name of the dataset", default="adult_income", choices=["adult_income"])
parser.add_argument('--run_name', type=str, help="Name to identify the run", required=True)
parser.add_argument('--epochs', type=int, help="Number of epochs for the gradient descent", default=30)
parser.add_argument('--learning_rate', type=float, help="Learning rate", default=1e-3)
parser.add_argument('--weight_decay', type=float, help="Weight decay", default=1e-5)
parser.add_argument('--warm_start_sample_size', type=int, help="Sample size for warm start", default=50)
parser.add_argument('--budget', type=int, help="Total budget", default=2000)
parser.add_argument('--k_nearby', type=int, help="Total nearby to choose", default=100)
parser.add_argument('--kernel', type=str, help="Kernel to choose", default='dot', choices=['dot', 'rbf'])
parser.add_argument('--noise', type=float, help="Noise for GP", default=0.1)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--num_iter', type=int, help="Number of iterations of sampling", default=20)
args = parser.parse_args()



def get_dataset(minmax_separate=False):
    dfx, dfy, cols = utils.get_dataset(args.dataset, return_dataframe=True)
    if minmax_separate:
        dfx_1 = dfx.loc[dfx.gender == 1]
        dfx_0 = dfx.loc[dfx.gender == 0]
        dfx_1[cols] = minmax_scale(dfx_1[cols])
        dfx_0[cols] = minmax_scale(dfx_0[cols])
    else:
        dfx[cols] = minmax_scale(dfx[cols])
        dfx_1 = dfx.loc[dfx.gender == 1]
        dfx_0 = dfx.loc[dfx.gender == 0]
        #dfy_1 = dfy.loc[dfx.gender == 1]
        #dfy_0 = dfy.loc[dfx.gender == 0]
    #return dfx_0, dfy_0, dfx_1, dfy_1
    return dfx_0, dfx_1

def process_blackbox(x):
    y = blackbox(x)
    y[y <= 0.5] = 0
    y[y > 0.5] = 1
    return y.long()


def generate_data(dfx_0, dfx_1, blackbox, sample_size):
        x0 = torch.tensor(dfx_0.values).to(device)
       # y0 = torch.tensor(dfy_0.values).to(device)
        x1 = torch.tensor(dfx_1.values).to(device)
       # y1 = torch.tensor(dfy_1.values).to(device)
        warm_start_samples_0 = utils.sample_from_tensor(x0.shape[0], sample_size, device).long()
        warm_start_samples_1 = utils.sample_from_tensor(x1.shape[0], sample_size, device).long()
        non_warm_start_samples_0 = utils.complement_idx(warm_start_samples_0, x0.shape[0])
        non_warm_start_samples_1 = utils.complement_idx(warm_start_samples_1, x1.shape[0])
       # warm_start_x0, warm_start_y0 = x0[warm_start_samples_0], y0[warm_start_samples_0]
       # warm_start_x1, warm_start_y1 = x1[warm_start_samples_1], y1[warm_start_samples_1]
        warm_start_x0 = x0[warm_start_samples_0]
        warm_start_x1 = x1[warm_start_samples_1]
        x0 = x0[non_warm_start_samples_0]
        x1 = x1[non_warm_start_samples_1]
        return x0, process_blackbox(x0), x1, process_blackbox(x1), warm_start_x0, process_blackbox(warm_start_x0), warm_start_x1, process_blackbox(warm_start_x1)



def iteration(kernel):
    x0, y0, x1, y1, cx0, cy0, cx1, cy1 = generate_data(dfx_0, dfx_1, blackbox, args.warm_start_sample_size)
    remaining = args.budget - cx0.shape[0]
    initial_disparity = utils.compute_disparity(cx0, cx1, True, blackbox)
    disparities = torch.zeros(2 + (args.budget - args.warm_start_sample_size)//args.k_nearby)
    disparities[0] = initial_disparity
    file_logger.info(f'Disparity @ {cx0.size()[0]}/{args.budget}: {round(initial_disparity.item(), 3)}')
    for epoch_outer in tqdm(range((args.budget - args.warm_start_sample_size)//args.k_nearby + 1)):
        # s = utils.sample_randomly_from_input_space('adult_income').to(device) 
        x_prime = SimpleNN(102,device).to(device)
        early_stopping = utils.EarlyStopping()
        # optimizer0 = torch.optim.AdamW((s,), lr=1e-5)
        optimizer = torch.optim.Adam(x_prime.parameters(),
                                1e-3,
                                weight_decay=1e-5)
        best_loss = 10e5
        count = 0
        losses = []
        
        for epoch in range(1, args.epochs):
            optimizer.zero_grad()
            #var_term = variance(x_prime(), cx0 ,device) + variance(x_prime(), cx1, device) 
            mean0, var0 = variance(cx0, x_prime(), cy0.double(), args.noise, kernel, device=device)
            mean1, var1 = variance(cx1, x_prime(), cy1.double(), args.noise, kernel, device=device)
            
            loss = -(var0 + var1)
            loss.backward()
            optimizer.step()
            early_stopping(loss, None)

       
        if remaining < args.k_nearby:
            qx0, qy0, x0, y0 = utils.query_nearby(x_prime(), x0, y0, remaining)
            qx1, qy1, x1, y1 = utils.query_nearby(x_prime(), x1, y1, remaining)
        else:
            qx0, qy0, x0, y0 = utils.query_nearby(x_prime(), x0, y0, args.k_nearby)
            qx1, qy1, x1, y1 = utils.query_nearby(x_prime(), x1, y1, args.k_nearby)
        remaining -= args.k_nearby
        cx0 = torch.cat((cx0, qx0), dim=0)
        cx1 = torch.cat((cx1, qx1), dim=0)
        cy0 = torch.cat((cy0, qy0), dim=0)
        cy1 = torch.cat((cy1, qy1), dim=0)
        current_disparity = utils.compute_disparity(cx0, cx1, True, blackbox)
        disparities[epoch_outer+1] = current_disparity
        print(f'Disparity @ {cx0.size()[0]}/{args.budget}: {round(current_disparity.item(), 3)}')
    return disparities.unsqueeze(0)




if __name__ == '__main__':
    utils.set_seed(42)
    torch.set_default_tensor_type(torch.DoubleTensor)
    log_dir = f'logs/run_logs/{args.dataset}'
    log_file = f'{log_dir}/{args.run_name}_training.log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok = True)
    file_logger = get_logger('Query', log_file)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    file_logger.info(f'Device: {device}')
    #dfx_0, dfy_0, dfx_1, dfy_1 = get_dataset()
    dfx_0, dfx_1 = get_dataset()

    if args.kernel == 'dot':
        kernel = dot_kernel
    elif args.kernel == 'rbf':
        kernel = rbf_kernel

    blackbox = BlackBox('Logistic', 102, 1).to(device)
    blackbox.load_state_dict(torch.load("checkpoints/adult_income/blackbox/Logistic/best.pt"))
    blackbox.eval()

    for i in range(args.num_iter):
        file_logger.info(f'Iteration {i+1}/{args.num_iter}')
        if i == 0:
            disp = iteration(kernel)
        else:
            d = iteration(kernel)
            disp = torch.cat((disp, d), dim=0)
    
    avg = disp.mean(dim=0)
    # Plot
    inputs = torch.zeros_like(avg)
    inputs[0] = args.warm_start_sample_size
    for i in range(1, len(inputs)-1):
        inputs[i] = inputs[i-1] + args.k_nearby
    inputs[-1] = args.budget - inputs[-2]

    total_parity = utils.compute_disparity(torch.tensor(dfx_0.values).to(device), torch.tensor(dfx_1.values).to(device), True, blackbox)

    fig, ax = plt.subplots()
    ax.plot(inputs, avg.detach().numpy(), label='variance', linewidth=2, marker='s')
    ax.plot(inputs, [total_parity.item()] * len(inputs), label='True', linewidth=2, marker='o')
    ax.grid()

    plt.savefig(f'plots/{args.run_name}.png')

    plt.show()












    






