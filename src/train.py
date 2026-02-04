# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from omniglot_dataset import OmniglotDataset
from tlu_dataset import TLUStatesDataset
from protonet import ProtoNet
from parser_util import get_parser

from tqdm import tqdm
import numpy as np
import torch
import os
import shutil
import time

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def init_dataset(opt, mode):
    if opt.dataset == 'tlu':
        dataset = TLUStatesDataset(mode=mode, root=opt.dataset_root)
    else:
        dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset

def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)

def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                             num_workers=4, pin_memory=True)
    return dataloader

def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    # Check if dataset is TLU (RGB) or Omniglot (Grayscale)
    x_dim = 3 if opt.dataset == 'tlu' else 1
    model = ProtoNet(x_dim=x_dim).to(device)
    return model

def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)

def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)

def save_checkpoint(state, is_best, checkpoint_dir, best_model_dir):
    """
    Saves checkpoint to disk
    """
    filepath = os.path.join(checkpoint_dir, 'last_model.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(best_model_dir, 'best_model.pth'))

def log_to_file(log_file, msg):
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    
    # State tracking
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0.0
    best_epoch = 0
    start_epoch = 0

    # Paths
    # Local output for fast IO (e.g., ./output on Colab content)
    # Drive output for backup (e.g., /content/drive/MyDrive/...)
    
    # We will assume opt.experiment_root is the DRIVE path (safe storage)
    # We create a local path for training speed
    local_exp_root = './checkpoints' 
    if not os.path.exists(local_exp_root):
        os.makedirs(local_exp_root)
        
    log_file_path = os.path.join(local_exp_root, 'log.txt')
    drive_log_path = os.path.join(opt.experiment_root, 'log.txt')
    
    # Check resume from DRIVE
    drive_checkpoint = os.path.join(opt.experiment_root, 'last_model.pth')
    
    if os.path.isfile(drive_checkpoint):
        print(f"==> Resuming from checkpoint: {drive_checkpoint}")
        checkpoint = torch.load(drive_checkpoint, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)
        model.load_state_dict(checkpoint['model_state'])
        optim.load_state_dict(checkpoint['optimizer_state'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
        print(f"==> Loaded checkpoint (epoch {checkpoint['epoch']})")
        
        # Restore logs if easy, otherwise just append
    else:
        print("==> No checkpoint found. Starting from scratch.")
        # Init log file header
        if not os.path.exists(log_file_path):
            log_to_file(log_file_path, "Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Is Best")

    for epoch in range(start_epoch, opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        
        current_train_loss = []
        current_train_acc = []
        
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            current_train_loss.append(loss.item())
            current_train_acc.append(acc.item())
            
        avg_train_loss = np.mean(current_train_loss)
        avg_train_acc = np.mean(current_train_acc)
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)
        
        lr_scheduler.step()
        
        # Validation
        avg_val_loss = 0.0
        avg_val_acc = 0.0
        is_best = False
        
        if val_dataloader is not None:
            val_iter = iter(val_dataloader)
            model.eval()
            current_val_loss = []
            current_val_acc = []
            
            with torch.no_grad():
                for batch in val_iter:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    model_output = model(x)
                    loss, acc = loss_fn(model_output, target=y,
                                        n_support=opt.num_support_val)
                    current_val_loss.append(loss.item())
                    current_val_acc.append(acc.item())
            
            avg_val_loss = np.mean(current_val_loss)
            avg_val_acc = np.mean(current_val_acc)
            val_loss.append(avg_val_loss)
            val_acc.append(avg_val_acc)
            
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                best_epoch = epoch
                best_state = model.state_dict()
                is_best = True
                
        # Logging
        log_str = f"{epoch:^5} | {avg_train_loss:.4f}     | {avg_train_acc:.4f}    | {avg_val_loss:.4f}   | {avg_val_acc:.4f}  | {str(is_best):^7}"
        print(log_str)
        log_to_file(log_file_path, log_str)

        # Checkpointing
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optim.state_dict(),
            'scheduler_state': lr_scheduler.state_dict(),
            'best_acc': best_acc,
            'best_epoch': best_epoch
        }
        
        # Save to local (FAST)
        save_checkpoint(state, is_best, local_exp_root, local_exp_root)
        
        # Backup to Drive every 9 epochs OR if best model found (to be safe)
        if is_best or epoch % 9 == 0:
            try:
                if not os.path.exists(opt.experiment_root):
                    os.makedirs(opt.experiment_root)
                    
                # Copy last_model
                shutil.copy(os.path.join(local_exp_root, 'last_model.pth'), 
                           os.path.join(opt.experiment_root, 'last_model.pth'))
                
                # Copy best_model if needed
                if is_best:
                    shutil.copy(os.path.join(local_exp_root, 'best_model.pth'), 
                               os.path.join(opt.experiment_root, 'best_model.pth'))
                               
                # Copy log
                shutil.copy(log_file_path, drive_log_path)
                print("==> Backed up checkpoint and logs to Drive.")
            except Exception as e:
                print(f"!! Warning: Backup to Drive failed: {e}")

    return best_state, best_acc, best_epoch

def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc

def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    
    # Ensure options.experiment_root exists (Drive path)
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')
    test_dataloader = init_dataloader(options, 'test')

    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
                
    best_state, best_acc, best_epoch = res
    
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    if best_state is not None:
        model.load_state_dict(best_state)
        print('Testing with best model.. (Epoch {})'.format(best_epoch))
        test(opt=options,
             test_dataloader=test_dataloader,
             model=model)

if __name__ == '__main__':
    main()
