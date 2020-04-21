import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import glob
import json
import pandas as pd

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config,config_f):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    input_size = list(data_loader.dataset[0][0].shape)[0]
    output_size = list(data_loader.dataset[0][1].shape)[0]
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, input_dim=input_size, output_dim=output_size)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
#     save result to JSON

    with np.load('results.npz') as result:
        mse,reg_binary_pred, F_1_score = [result[i] for i in ('mse','regression_binary_pred', 'F_1_score')]

    with open(config_f, "r") as f_object:
        data = json.load(f_object)

    data["results"]["mse"] = np.sum(mse)
    data["results"]["accuracy"] = np.sum(reg_binary_pred)
    data["results"]["f-1 score"] = np.sum(F_1_score)
    with open(config_f, "w") as f_object:
        json.dump(data, f_object,indent=4)


def save_to_excel():
    name_list = []
    batch_size = []
    context_win = []
    input_columns = []
    target_columns = []
    regression_binary_pred = []
    F_1_score = []
    mse = []

    for fname in glob.glob('data_loader/configs/*.json'):
        with open(fname, "r") as f_object:
            data = json.load(f_object)

            name_list.append(fname.split("\\")[1].split(".")[0])
            batch_size.append(data["data_loader"]["args"]["batch_size"])
            context_win.append(data["data_loader"]["args"]["window"])
            input_columns.append(data["data_loader"]["args"]["input_columns"])
            target_columns.append(data["data_loader"]["args"]["target_columns"])
            regression_binary_pred.append(data["results"]["accuracy"])
            F_1_score.append(data["results"]["f-1 score"])
            mse.append(data["results"]["mse"])

    df = pd.DataFrame(np.array([name_list,batch_size,context_win,input_columns,target_columns,regression_binary_pred,F_1_score,mse]).T,
                      columns=["file names","batch size","context window","input columns","target_columns","regression_binary_pred","F_1_score","mse"])

    print(df)

    df.to_excel('data_loader/configs/saved_results.xlsx')

if __name__ == '__main__':

    # write your for loop in here
    # config = ConfigParser.from_args(args,["-c","config.json"],options)
    # main(config)



    for fname in glob.glob('data_loader/configs/*.json'):
        print("loading... ",fname)
        args = argparse.ArgumentParser(description='PyTorch Template')
        args.add_argument('-c', '--config', default=None, type=str,
                          help='config file path (default: None)')
        args.add_argument('-r', '--resume', default=None, type=str,
                          help='path to latest checkpoint (default: None)')
        args.add_argument('-d', '--device', default=None, type=str,
                          help='indices of GPUs to enable (default: all)')

        # custom cli options to modify configuration from default values given in json file.
        CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
        options = [
            CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
            CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
        ]
        config = ConfigParser.from_args(args, ["-c", fname], options)
        main(config,fname)

    save_to_excel()

    # save all config files into excel
