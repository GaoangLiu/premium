import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import codefast as cf 

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

if args.save_model:
    cf.info('save it')
