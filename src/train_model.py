import sys
import time
import random
import os
import config
from math import inf
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils import calculate_metric
from utils import writer, make_optimizer_and_scheduler
from model import set_parameter_requires_grad, initialize_model, get_hook_names, SimpleCLF
import operator
from attention_mask import attention_gen_patches

sys.path.insert(0, '../src')

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)
device = config.DEVICE


def train_model(train_loader, val_loader, criterions, num_epochs=config.NUM_EPOCHS,
                save_freq=5, verbose=config.VERBOSE):
    start_time = time.time()
    print(f"    Mode          : {device}")

    # initialize models
    g_model, g_input_size, _, g_fm_name, g_pool_name, g_fm_size = initialize_model(config.GLOBAL_MODEL_NAME)
    l_model, l_input_size, _, _, l_pool_name, l_fm_size = initialize_model(config.LOCAL_MODEL_NAME)
    f_feature_size = g_fm_size + l_fm_size
    if config.USE_EXTRA_INPUT:
        f_feature_size += 3
    f_model = SimpleCLF(input_size=f_feature_size).to(device)

    g_criterion, l_criterion, f_criterion = criterions
    print(f"    Global Model type    : {type(g_model)}")
    print(f"    Local  Model type    : {type(l_model)}")
    print(f"    Fusion Model type    : {type(f_model)}")

    # prepare hooks for branches
    g_activation = get_hooks(g_model, config.GLOBAL_MODEL_NAME)
    l_activation = get_hooks(l_model, config.LOCAL_MODEL_NAME)

    # initialize optimizer and scheduler
    g_optimizer, g_scheduler = make_optimizer_and_scheduler(g_model, lr=config.GLOBAL_LEARNING_RATE)
    l_optimizer, l_scheduler = make_optimizer_and_scheduler(l_model, lr=config.LOCAL_LEARNING_RATE)
    f_optimizer, f_scheduler = make_optimizer_and_scheduler(f_model, lr=config.FUSION_LEARNING_RATE, patience=5)

    train_losses, val_losses, val_rocs = [], [], []
    best_val_loss, roc_at_best_val_loss = inf, 0.0
    best_model_paths = ['', '', '']
    best_epoch = 1
    cudnn.benchmark = True
    print(f"Training started")
    for epoch in range(1, num_epochs+1):  # loop over the dataset multiple times
        print(f"\nEpoch {epoch}")
        epoch_start_time = time.time()
        g_model.train()
        l_model.train()
        f_model.train()
        # tune cls layers for a few epochs, then tune the whole model
        if config.FINE_TUNE and epoch == config.FINE_TUNE_START_EPOCH:
            set_parameter_requires_grad(g_model, feature_extracting=False)
            set_parameter_requires_grad(l_model, feature_extracting=False)
            # update optimizer and scheduler to tune all parameter groups
            g_optimizer, g_scheduler = make_optimizer_and_scheduler(g_model, lr=config.GLOBAL_LEARNING_RATE)
            l_optimizer, l_scheduler = make_optimizer_and_scheduler(l_model, lr=config.GLOBAL_LEARNING_RATE)
            # decrease learning rate
            for param_group in g_optimizer.param_groups:
                param_group['lr'] /= 5
            for param_group in l_optimizer.param_groups:
                param_group['lr'] /= 5
            if verbose:
                print("Starting fine-tuning all model parameters")

        g_running_loss, l_running_loss, f_running_loss = 0.0, 0.0, 0.0
        iterations = len(train_loader)
        for i, (inputs, labels) in enumerate(train_loader, 0):
            bz = inputs.shape[0]
            # similar to optimizer.zero_grad().
            # https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/6
            g_model.zero_grad()
            l_model.zero_grad()
            f_model.zero_grad()

            # get the inputs: a list of [inputs, labels]
            # `labels` is an 14-dim list of tensors
            inputs, labels = inputs.to(device), labels.to(device)

            ####################################
            # Global Branch
            ####################################
            g_outputs = g_model(inputs).to(device)
            ####################################
            # Local branch
            ####################################
            fm_global = g_activation['fm']  # get feature map
            local_patches = attention_gen_patches(inputs, fm_global)
            l_outputs = l_model(local_patches).to(device)
            ####################################
            # Fusion branch
            ####################################
            pool_g = g_activation['pool'].view(bz, -1)
            pool_l = l_activation['pool'].view(bz, -1)
            #print(f"Global pooling size: {pool_g.shape}")
            #print(f"Local pooling size: {pool_l.shape}")
            # TODO: stack extra features
            if config.USE_EXTRA_INPUT:
                extra = torch.rand(bz, 3)  # FIXME
            else:
                extra = None
            f_outputs = f_model(pool_g, pool_l, extra).to(device)

            # compute loss
            g_loss = g_criterion(g_outputs, labels)
            l_loss = l_criterion(l_outputs, labels)
            f_loss = f_criterion(f_outputs, labels)

            # back-propagate
            g_loss.backward()
            l_loss.backward()
            f_loss.backward()

            g_running_loss += float(g_loss.item())
            l_running_loss += float(l_loss.item())
            f_running_loss += float(f_loss.item())

            # step optimizer
            g_optimizer.step()
            l_optimizer.step()
            f_optimizer.step()

            # print statistics
            if verbose:
                print(".", end="")
                if (i+1) % 50 == 0:  # print every 50 mini-batches
                    print(f' Epoch: {epoch:>2} '
                          f' Batch: {i + 1:>3} / {iterations} '
                          f' Global loss: {g_running_loss / (i + 1):>6.4f} '
                          f' Local  loss: {l_running_loss / (i + 1):>6.4f} '
                          f' Fusion loss: {f_running_loss / (i + 1):>6.4f} '
                          f' Average batch time: {(time.time() - epoch_start_time) / (i+1):>4.3f} secs ')

        print(f"\nTraining loss: {f_running_loss / len(train_loader):>4f}")
        train_losses.append(f_running_loss / len(train_loader))  # keep trace of train loss in each epoch

        # # write training loss to TensorBoard
        writer.add_scalar("Loss/g_train", g_running_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/l_train", l_running_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/f_train", f_running_loss / len(train_loader), epoch)

        print(f'Time elapsed: {(time.time()-start_time)/60.0:.1f} minutes.')

        if epoch % save_freq == 0:  # save model and checkpoint for inference or training
            save_models([g_model, l_model, f_model], epoch)
            save_checkpoints([g_model, l_model, f_model],
                             epoch,
                             [g_optimizer, l_optimizer, f_optimizer],
                             [g_running_loss, l_running_loss, f_running_loss])

        ##################################################################
        # Validation
        ##################################################################
        print("Validating...")
        v_loss, v_aucs, y_probs, _, y_true = eval_models([g_model, l_model, f_model], val_loader, criterions)

        val_losses.append(v_loss)
        val_rocs.append(v_aucs)

        # save model if best ROC on fusion branch
        if v_loss[2] < best_val_loss:
            best_epoch = epoch
            best_val_loss = v_loss[2]
            roc_at_best_val_loss = v_aucs[2]
            best_model_paths = save_models([g_model, l_model, f_model], epoch, best=True)

        writer.add_scalar("Loss/g_val", v_loss[0], epoch)
        writer.add_scalar("ROCAUC/g_val", v_aucs[0], epoch)
        writer.add_pr_curve('PR/g_val', y_true, y_probs[0], epoch)

        writer.add_scalar("Loss/l_val", v_loss[1], epoch)
        writer.add_scalar("ROCAUC/l_val", v_aucs[1], epoch)
        writer.add_pr_curve('PR/l_val', y_true, y_probs[1], epoch)

        writer.add_scalar("Loss/f_val", v_loss[2], epoch)
        writer.add_scalar("ROCAUC/f_val", v_aucs[1], epoch)
        writer.add_pr_curve('PR/f_val', y_true, y_probs[2], epoch)

        # step the learning rate
        if g_scheduler is not None:
            g_scheduler.step(v_loss[0])
        if l_scheduler is not None:
            l_scheduler.step(v_loss[1])
        if f_scheduler is not None:
            f_scheduler.step(v_loss[2])

        writer.flush()
        print(f'Time elapsed: {(time.time() - start_time) / 60.0:.1f} minutes.')
        print(f'Average time per epoch: {(time.time() - start_time) / 60.0/ epoch:.1f} minutes.')
        # early stopping
        if epoch - best_epoch > config.EARLY_STOP_EPOCHS:
            print(f"No improvement for {config.EARLY_STOP_EPOCHS} epochs. Stop training.")
            break

    print(f'Finished Training. Total time: {(time.time() - start_time) / 60} minutes.')
    print(f"Best (fusion branch) ROC on validation set: {best_val_loss:3f}, achieved on epoch #{best_epoch}")

    return [g_model, l_model, f_model], \
           train_losses, val_losses, best_val_loss, val_rocs, roc_at_best_val_loss, best_model_paths


def eval_models(models,
                loader,
                criterions,
                g_model_name=config.GLOBAL_MODEL_NAME,
                l_model_name=config.LOCAL_MODEL_NAME,
                threshold=0.50,
                verbose=config.VERBOSE):
    g_model, l_model, f_model = models
    g_criterion, l_criterion, f_criterion = criterions
    g_activation = get_hooks(g_model, g_model_name)
    l_activation = get_hooks(l_model, l_model_name)

    y_true = []
    g_y_prob, g_y_pred = [], []
    l_y_prob, l_y_pred = [], []
    f_y_prob, f_y_pred = [], []
    g_loss, l_loss, f_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        g_model.eval()
        l_model.eval()
        f_model.eval()
        cudnn.benchmark = True
        for i, (inputs, labels) in enumerate(loader):
            bz = inputs.shape[0]
            labels = labels.to(config.DEVICE)  # true labels
            y_true.append(labels)

            # Global branch
            g_probs = g_model(inputs.to(config.DEVICE))  # pass through
            g_loss += float(torch.mean(g_criterion(g_probs, labels)))
            g_predicted = g_probs > threshold
            g_y_prob.append(g_probs)
            g_y_pred.append(g_predicted)

            # Local branch
            fm_global = g_activation['fm']  # get feature map
            local_patches = attention_gen_patches(inputs, fm_global)
            l_probs = l_model(local_patches).to(device)
            l_loss += float(torch.mean(l_criterion(l_probs, labels)))
            l_predicted = l_probs > threshold
            l_y_prob.append(l_probs)
            l_y_pred.append(l_predicted)

            # Fusion branch
            pool_g = g_activation['pool'].view(bz, -1)
            pool_l = l_activation['pool'].view(bz, -1)
            # TODO: stack extra features
            if config.USE_EXTRA_INPUT:
                extra = torch.zeros(bz, 3)
            else:
                extra = None
            f_probs = f_model(pool_g, pool_l, extra).to(device)
            f_loss += float(torch.mean(f_criterion(f_probs, labels)))
            f_predicted = f_probs > threshold
            f_y_prob.append(f_probs)
            f_y_pred.append(f_predicted)

    g_loss /= len(loader)
    l_loss /= len(loader)
    f_loss /= len(loader)

    # convert to numpy
    g_y_prob = torch.cat(g_y_prob).detach().cpu().numpy()
    g_y_pred = torch.cat(g_y_pred).detach().cpu().numpy()
    l_y_prob = torch.cat(l_y_prob).detach().cpu().numpy()
    l_y_pred = torch.cat(l_y_pred).detach().cpu().numpy()
    f_y_prob = torch.cat(f_y_prob).detach().cpu().numpy()
    f_y_pred = torch.cat(f_y_pred).detach().cpu().numpy()
    y_true = torch.cat(y_true).detach().cpu().numpy()

    g_macro_roc_auc, g_roc_aucs = calculate_metric(g_y_prob, y_true)
    l_macro_roc_auc, l_roc_aucs = calculate_metric(l_y_prob, y_true)
    f_macro_roc_auc, f_roc_aucs = calculate_metric(f_y_prob, y_true)

    # print results
    if verbose:
        print("'"*20)
        print("Global branch")
        print("'" * 20)
        print(f"Disease:{'':<22}ROCAUC")
        for i, lb in enumerate(config.TEXT_LABELS):
            print(f"{lb:<30}: {g_roc_aucs[i]:.4f}")
        print(f"\nROCAUC (Macro): {g_macro_roc_auc}")

        print("'" * 20)
        print("Local branch")
        print("'" * 20)
        print(f"Disease:{'':<22}ROCAUC")
        for i, lb in enumerate(config.TEXT_LABELS):
            print(f"{lb:<30}: {l_roc_aucs[i]:.4f}")
        print(f"\nROCAUC (Macro): {l_macro_roc_auc}")

        print("'" * 20)
        print("Fusion branch")
        print("'" * 20)
        print(f"Disease:{'':<22}ROCAUC")
        for i, lb in enumerate(config.TEXT_LABELS):
            print(f"{lb:<30}: {f_roc_aucs[i]:.4f}")
        print(f"\nROCAUC (Macro): {f_macro_roc_auc}")

    return [g_loss, l_loss, f_loss], \
           [g_macro_roc_auc, l_macro_roc_auc, f_macro_roc_auc], \
           [g_y_prob, l_y_prob, f_y_prob], \
           [g_y_pred, l_y_pred, f_y_pred], \
           y_true


def save_models(models, num_epochs, root_dir=config.ROOT_PATH, model_dir=config.MODEL_DIR, best=False, verbose=True):
    g_model, l_model, f_model = models

    model_subdir = 'model_' + config.WRITER_NAME.split('/')[-1]
    folder_path = os.path.join(root_dir, model_dir, model_subdir)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if best:
        g_path = os.path.join(folder_path, f'{config.GLOBAL_MODEL_NAME}_best.pth')
        l_path = os.path.join(folder_path, f'{config.LOCAL_MODEL_NAME}_best.pth')
        f_path = os.path.join(folder_path, f'{config.FUSION_MODEL_NAME}_best.pth')
    else:
        g_path = os.path.join(folder_path, f'{config.GLOBAL_MODEL_NAME}_{num_epochs}epoch.pth')
        l_path = os.path.join(folder_path, f'{config.LOCAL_MODEL_NAME}_{num_epochs}epoch.pth')
        f_path = os.path.join(folder_path, f'{config.FUSION_MODEL_NAME}_{num_epochs}epoch.pth')

    torch.save(g_model.state_dict(), g_path)
    torch.save(l_model.state_dict(), l_path)
    torch.save(f_model.state_dict(), f_path)
    return [g_path, l_path, f_path]


def save_checkpoints(models, epoch, optimizers, losses):
    g_model, l_model, f_model = models
    g_optimizer, l_optimizer, f_optimizer = optimizers
    g_loss, l_loss, f_loss = losses

    root_dir = config.ROOT_PATH
    ckpt_dir = config.CHECKPOINT_DIR
    folder_path = os.path.join(root_dir, ckpt_dir)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    g_ckpt_subdir = 'g_checkpoint_' + config.WRITER_NAME.split('/')[-1]
    l_ckpt_subdir = 'l_checkpoint_' + config.WRITER_NAME.split('/')[-1]
    f_ckpt_subdir = 'f_checkpoint_' + config.WRITER_NAME.split('/')[-1]

    g_path = os.path.join(folder_path, g_ckpt_subdir)
    l_path = os.path.join(folder_path, l_ckpt_subdir)
    f_path = os.path.join(folder_path, f_ckpt_subdir)

    torch.save({
        'epoch': epoch,
        'model_state_dict': g_model.state_dict(),
        'optimizer_state_dict': g_optimizer.state_dict(),
        'loss': g_loss,
    }, g_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': l_model.state_dict(),
        'optimizer_state_dict': l_optimizer.state_dict(),
        'loss': l_loss,
    }, l_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': f_model.state_dict(),
        'optimizer_state_dict': f_optimizer.state_dict(),
        'loss': f_loss,
    }, f_path)
    print(f"Checkpoint saved.")
    return None


def get_hooks(model, model_name):
    # prepare hook for branches
    activation = dict()

    def get_activation(name, act):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    fm_name, pool_name = get_hook_names(model_name)

    # feature map and pooling of global model, pooling of local model
    activation['fm'] = operator.attrgetter(fm_name)(model).register_forward_hook(get_activation('fm', activation))
    activation['pool'] = operator.attrgetter(pool_name)(model).register_forward_hook(get_activation('pool', activation))
    return activation
