import random
import numpy as np
import torch
import torch.backends.cudnn
from examples.tabular.tabular import main

if __name__ == '__main__':
    # Deterministic Behavior
    seed = 0
    torch.cuda.set_device(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()