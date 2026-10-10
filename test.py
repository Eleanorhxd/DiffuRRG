import torch
import argparse
import numpy as np
import os
import random
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.model import DiffuRRG
from modules.utils import parse_args, auto_resume_helper, load_checkpoint
from modules.logger import create_logger
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def main():
    # parse arguments
    args, config = parse_args()
    print(args)

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(config.OUTPUT, exist_ok=True)
    #logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    logger = create_logger(output_dir=config.OUTPUT,  name=f"{config.MODEL.NAME}")

    #if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    logger.info(config.dump())


    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = DiffuRRG(args, tokenizer, logger, config)

    resume = False
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(config, optimizer, len(train_dataloader))


    state_dict = torch.load(args.pretrained)['state_dict']
    logger.info(state_dict.keys())
    model.load_state_dict(state_dict, strict=True)
    logger.info(f'loading pretrained model {args.pretrained}, ignoring auto resume')

    model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    print('SWR2Gen Transformer Training')

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, tokenizer,
                      train_dataloader, val_dataloader, test_dataloader, writer, logger, config)
    trainer._test()
    writer.close()



if __name__ == '__main__':
    main()
