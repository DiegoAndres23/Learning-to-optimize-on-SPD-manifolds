import torch
import torch.nn as nn

from learning_to_learn import Learning_to_learn_global_training
from LSTM_Optimizee_Model import LSTM_Optimizee_Model
from hand_optimizer.handcraft_optimizer import Hand_Optimizee_Model
from DataSet.KYLBERG import KYLBERG
import config

# Forzar CPU
device = torch.device("cpu")

opt = config.parse_opt()
opt.batchsize_para = opt.category_num * opt.sample_num
print(opt)

# Crear modelo en CPU
LSTM_Optimizee = LSTM_Optimizee_Model(
    opt, opt.DIM, opt.DIM, opt.DIM,
    batchsize_data=opt.batchsize_data,
    batchsize_para=opt.batchsize_para
).to(device)

# Cargar pesos preentrenados (si existen)
if opt.loadpretrain:
    checkpoint = torch.load(opt.prepath, map_location=device)
    LSTM_Optimizee.load_state_dict(checkpoint, strict=False)

    checkpoint2 = torch.load(opt.prepath2, map_location=device)
    LSTM_Optimizee.load_state_dict(checkpoint2, strict=False)

Hand_Optimizee = Hand_Optimizee_Model(opt.hand_optimizer_lr).to(device)

print('Pretrain finished')

# Dataset y DataLoader
train_mnist = KYLBERG(opt.datapath, train=True)
train_loader = torch.utils.data.DataLoader(
    train_mnist,
    batch_size=opt.batchsize_data,
    shuffle=True,
    drop_last=False,
    num_workers=0
)

# Entrenamiento global (en CPU)
global_loss_list, flag = Learning_to_learn_global_training(
    opt,
    Hand_Optimizee,
    LSTM_Optimizee,
    train_loader
)
