import yaml
import pickle
import numpy as np
from model import MASTER
from trainer import Trainer
import torch
from pathlib import Path
import logging
import argparse

def load_config(cfg):
    file= f'configs/{cfg}.yaml'
    with open(file, 'r') as f:
        return yaml.safe_load(f)
 
def load_data(config):
    data_cfg = config['data']
    with open(f"{data_cfg['train_data_dir']}/{data_cfg['prefix']}/{data_cfg['universe']}_dl_train.pkl", 'rb') as f:
        dl_train = pickle.load(f)
    with open(f"{data_cfg['predict_data_dir']}/{data_cfg['universe']}_dl_valid.pkl", 'rb') as f:
        dl_valid = pickle.load(f)
    with open(f"{data_cfg['predict_data_dir']}/{data_cfg['universe']}_dl_test.pkl", 'rb') as f:
        dl_test = pickle.load(f)
    return dl_train, dl_valid, dl_test

def init_logging(cfg):
    logging_path = f'logs/{cfg}.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(logging_path),  # 日志保存到文件
            logging.StreamHandler()  # 同时输出到控制台
        ],
        force=True
    )
    return str(logging_path)


def main(cfg: str):
    
    config = load_config(cfg)
    init_logging(cfg)
    if config['data']['universe'] == 'csi300':
        config['model']['beta'] = 5
    elif config['data']['universe'] == 'csi800':
        config['model']['beta'] = 2
    
    dl_train, dl_valid, dl_test = load_data(config)
    
    ic, icir, ric, ricir = [], [], [], []
    
    for seed in config['training']['seeds']:
        config['training']['seed'] = seed
        trainer = Trainer(config)
        model = MASTER(config).to(trainer.device)
        
        logging.info(f"Training seed {seed}")
        trainer.fit(model, dl_train, dl_valid)
        
        param_path = f"{config['training']['save_path']}/{config['data']['universe']}_{seed}.pkl"
        model.load_state_dict(torch.load(param_path, map_location=trainer.device))
        _, metrics = trainer.test(model, trainer._dataloader(dl_test, shuffle=False))
        logging.info(f"Seed {seed}: {metrics}")
        
        ic.append(metrics['IC'])
        icir.append(metrics['ICIR'])
        ric.append(metrics['RIC'])
        ricir.append(metrics['RICIR'])
    
    logging.info(f"IC: {np.mean(ic):.4f} ± {np.std(ic):.4f}")
    logging.info(f"ICIR: {np.mean(icir):.4f} ± {np.std(icir):.4f}")
    logging.info(f"RIC: {np.mean(ric):.4f} ± {np.std(ric):.4f}")
    logging.info(f"RICIR: {np.mean(ricir):.4f} ± {np.std(ricir):.4f}")

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg1') 
    args = parser.parse_args()
    main(args.cfg)


'''
conda activate master
cd /home/yzfang/workplace/out_proj/master2
python3 main.py --cfg cfg1_csi300

'''










