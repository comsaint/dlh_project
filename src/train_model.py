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
from model import initialize_model, get_hook_names, SimpleCLF
import operator
from attention_mask import attention_gen_patches
import torchvision.transforms.functional as F

sys.path.insert(0, '../src')

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)
device = config.DEVICE


def train_model(params, train_loader, val_loader, criterions, save_freq=5):
    start_time = time.time()
    print(f"    Mode          : {device}")

    # initialize models
    g_model, g_input_size, g_use_model_loss, g_fm_name, g_pool_name, g_fm_size = initialize_model(params, params['GLOBAL_MODEL_NAME'])
    l_model, l_input_size, l_use_model_loss, _, l_pool_name, l_fm_size = initialize_model(params, params['LOCAL_MODEL_NAME'])
    f_feature_size = g_fm_size + l_fm_size
    if params['USE_EXTRA_INPUT']:
        f_feature_size += 3
    f_model = SimpleCLF(input_size=f_feature_size).to(device)
    print(f"Fusion input size: {f_feature_size}")
    g_criterion, l_criterion, f_criterion = criterions
    print(f"    Global Model type    : {type(g_model)}")
    print(f"    Local  Model type    : {type(l_model)}")
    print(f"    Fusion Model type    : {type(f_model)}")

    # prepare hooks for branches
    g_activation = get_hooks(g_model, params['GLOBAL_MODEL_NAME'])
    l_activation = get_hooks(l_model, params['LOCAL_MODEL_NAME'])

    # initialize optimizer and scheduler
    g_optimizer, g_scheduler = make_optimizer_and_scheduler(g_model, lr=params['GLOBAL_LEARNING_RATE'])
    l_optimizer, l_scheduler = make_optimizer_and_scheduler(l_model, lr=params['LOCAL_LEARNING_RATE'])
    f_optimizer, f_scheduler = make_optimizer_and_scheduler(f_model, lr=params['FUSION_LEARNING_RATE'])

    train_losses, val_losses, val_rocs = [], [], []
    best_val_loss, roc_at_best_val_loss = inf, 0.0
    best_model_paths = ['', '', '']
    best_epoch = 1
    cudnn.benchmark = True
    torch.cuda._lazy_init()
    
    print(f"Training started")
    unfreeze_flag = False
    for epoch in range(1, params['NUM_EPOCHS']+1):  # loop over the dataset multiple times
        print(f"\nEpoch {epoch}")
        epoch_start_time = time.time()
        g_model.train()
        l_model.train()
        f_model.train()

        # tune cls layers for a few epochs, then tune the whole model
        if params["FINE_TUNE"] and epoch == params['FINE_TUNE_START_EPOCH']:
            if config.VERBOSE:
                print("Starting fine-tuning model parameters")
            if params["FINE_TUNE_STEP_WISE"] is False:
                if config.VERBOSE:
                    print("Unfreeze all parameters")
                for param in g_model.parameters():
                    param.requires_grad = True
                for param in l_model.parameters():
                    param.requires_grad = True
            else:
                if config.VERBOSE:
                    print("Unfreeze parameters stepwise")
                unfreeze_flag = True
            # update optimizer and scheduler to tune all parameter groups
            # TODO: are they inplace?
            g_optimizer, g_scheduler = make_optimizer_and_scheduler(g_model, lr=params['GLOBAL_LEARNING_RATE'] / 5)
            l_optimizer, l_scheduler = make_optimizer_and_scheduler(l_model, lr=params['LOCAL_LEARNING_RATE'] / 5)

        if unfreeze_flag:
            if config.VERBOSE:
                print("Unfreeze a layer...")
            # After FINE_TUNE_START_EPOCH, unfreeze last frozen layer(s) every epoch
            g_ufl = unfreeze_last_frozen_layer(g_model)
            l_ufl = unfreeze_last_frozen_layer(l_model)
            # if there is an unfreeze, update the optimizer and scheduler
            if g_ufl:
                g_optimizer, g_scheduler = make_optimizer_and_scheduler(g_model, lr=params['GLOBAL_LEARNING_RATE'] / 5)
            elif config.VERBOSE:
                print("No more layer to unfreeze in Global model.")
            if l_ufl:
                l_optimizer, l_scheduler = make_optimizer_and_scheduler(l_model, lr=params['LOCAL_LEARNING_RATE'] / 5)
            elif config.VERBOSE:
                print("No more layer to unfreeze in Local model.")

        if config.VERBOSE:
            g_all_params = sum([p.numel() for p in g_model.parameters()])
            l_all_params = sum([p.numel() for p in l_model.parameters()])
            f_all_params = sum([p.numel() for p in f_model.parameters()])
            g_train_params = sum(p.numel() for p in g_model.parameters() if p.requires_grad)
            l_train_params = sum(p.numel() for p in l_model.parameters() if p.requires_grad)
            f_train_params = sum(p.numel() for p in f_model.parameters() if p.requires_grad)
            g_perc = g_train_params / g_train_params * 100
            l_perc = l_train_params / l_train_params * 100
            f_perc = f_train_params / f_train_params * 100
            print(f"Number of trainable parameters in Global model: {g_train_params}; Total: {g_all_params}({g_perc}%)")
            print(f"Number of trainable parameters in Local model: {l_train_params}; Total: {l_all_params}({l_perc}%)")
            print(f"Number of trainable parameters in Fusion model: {f_train_params}; Total: {f_all_params}({f_perc}%)")

        g_running_loss, l_running_loss, f_running_loss = 0.0, 0.0, 0.0
        iterations = len(train_loader)
        for i, (inputs, labels, extra_features) in enumerate(train_loader, 0):
            bz, img_ch, _, _ = inputs.shape
            
            # similar to optimizer.zero_grad().
            # https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/6
            g_model.zero_grad()
            l_model.zero_grad()
            f_model.zero_grad()

            # get the inputs
            inputs, labels, extra_features = inputs.to(device), labels.to(device), extra_features.to(config.DEVICE)
            ####################################
            # Global Branch
            ####################################
            g_inputs = F.resize(inputs, g_input_size)
            g_outputs = g_model(g_inputs).to(device)
            ####################################
            # Local branch
            ####################################
            fm_global = g_activation['fm']  # get feature map
            local_patches = attention_gen_patches(inputs, fm_global, params['HEATMAP_THRESHOLD'])
            l_outputs = l_model(local_patches).to(device)

            ####################################
            # Fusion branch
            ####################################
            pool_g = g_activation['pool'].view(bz, -1)
            pool_l = l_activation['pool'].view(bz, -1)
            #print(f"Global pooling size: {pool_g.shape}")
            #print(f"Local pooling size: {pool_l.shape}")
            if not params['USE_EXTRA_INPUT']:
                extra_features = None
            #print(f"G size: {pool_g.shape[1]}, L size: {pool_l.shape[1]}, E_size: {extra_features.shape[1]}")
            f_outputs = f_model(pool_g, pool_l, extra_features).to(device)

            # compute loss
            if g_use_model_loss:
                reconstructions = None
                g_loss = g_model.loss(g_inputs, g_outputs, labels, mean_error=True, reconstruct=config.RECONSTRUCT)
                
                if type(g_loss) == tuple and len(g_loss) == 2:
                    reconstructions = g_loss[1]
                    g_loss = g_loss[0]
                '''
                # Reproduce the decoded image in Run Directory
                if i == 0 and reconstructions != None: 
                    folder_path = os.path.join(config.ROOT_PATH, config.OUT_DIR, config.MODEL_NAME)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    save_image(inputs         ,f'{folder_path}/epoch_{epoch}_01_original.png')
                    save_image(reconstructions,f'{folder_path}/epoch_{epoch}_02_reconstr.png')
            
                del reconstructions
                '''
            else:
                g_loss = g_criterion(g_outputs, labels)
                
            if l_use_model_loss:
                reconstructions = None
                l_loss = l_model.loss(local_patches, l_outputs, labels, mean_error=True, reconstruct=config.RECONSTRUCT)
                
                if type(l_loss) == tuple and len(l_loss) == 2:
                    reconstructions = l_loss[1]
                    l_loss = l_loss[0]

                '''
                # Reproduce the decoded image in Run Directory
                if i == 0 and reconstructions is not None: 
                    folder_path = os.path.join(config.ROOT_PATH, config.OUT_DIR, config.MODEL_NAME)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    save_image(inputs         ,f'{folder_path}/epoch_{epoch}_01_original.png')
                    save_image(reconstructions,f'{folder_path}/epoch_{epoch}_02_reconstr.png')
                
                del reconstructions
                '''
            else:
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
            if config.VERBOSE:
                print(".", end="")
                if (i+1) % 50 == 0:  # print every 50 mini-batches
                    print(f' Epoch: {epoch:>2} '
                          f' Batch: {i + 1:>4} / {iterations} '
                          f' Global loss: {g_running_loss / (i + 1):>6.4f} '
                          f' Local  loss: {l_running_loss / (i + 1):>6.4f} '
                          f' Fusion loss: {f_running_loss / (i + 1):>6.4f} '
                          f' Average batch time: {(time.time() - epoch_start_time) / (i+1):>4.3f} secs ')

        print(f"\nTraining loss: {f_running_loss / len(train_loader):>4f}")
        train_losses.append(f_running_loss / len(train_loader))  # keep trace of train loss in each epoch

        # write training loss to TensorBoard
        writer.add_scalar("Loss/g_train", g_running_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/l_train", l_running_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/f_train", f_running_loss / len(train_loader), epoch)

        print(f'Time elapsed: {(time.time()-start_time)/60.0:.1f} minutes.')

        if epoch % save_freq == 0:  # save model and checkpoint for inference or training
            save_models(params, [g_model, l_model, f_model], epoch)
            save_checkpoints([g_model, l_model, f_model],
                             epoch,
                             [g_optimizer, l_optimizer, f_optimizer],
                             [g_running_loss, l_running_loss, f_running_loss])

        ##################################################################
        # Validation
        ##################################################################
        print("Validating...")
        v_loss, v_aucs, y_probs, _, y_true = eval_models(params,
                                                         [g_model, l_model, f_model],
                                                         val_loader,
                                                         [g_criterion, l_criterion, f_criterion],
                                                         g_use_model_loss=g_use_model_loss,
                                                         l_use_model_loss=l_use_model_loss,
                                                         g_input_size=g_input_size,
                                                         l_input_size=l_input_size
                                                         )
        val_losses.append(v_loss)
        val_rocs.append(v_aucs)

        # save model if best ROC achieved on fusion branch
        if v_loss[2] < best_val_loss:
            best_epoch = epoch
            best_val_loss = v_loss[2]
            roc_at_best_val_loss = v_aucs[2]
            best_model_paths = save_models(params, [g_model, l_model, f_model], epoch, best=True)

        writer.add_scalar("Loss/g_val", v_loss[0], epoch)
        writer.add_scalar("ROCAUC/g_val", v_aucs[0], epoch)
        writer.add_pr_curve('PR/g_val', y_true, y_probs[0], epoch)

        writer.add_scalar("Loss/l_val", v_loss[1], epoch)
        writer.add_scalar("ROCAUC/l_val", v_aucs[1], epoch)
        writer.add_pr_curve('PR/l_val', y_true, y_probs[1], epoch)

        writer.add_scalar("Loss/f_val", v_loss[2], epoch)
        writer.add_scalar("ROCAUC/f_val", v_aucs[2], epoch)
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
        if epoch - best_epoch > params['EARLY_STOP_EPOCHS']:
            print(f"No improvement for {params['EARLY_STOP_EPOCHS']} epochs. Stop training.")
            break

    print(f'Finished Training. Total time: {(time.time() - start_time) / 60} minutes.')
    print(f"Best (fusion branch) ROC on validation set: {best_val_loss:3f}, achieved on epoch #{best_epoch}")

    return [g_model, l_model, f_model], train_losses, val_losses, best_val_loss, val_rocs, roc_at_best_val_loss, best_model_paths


def eval_models(params,
                models,
                loader,
                criterions,
                g_use_model_loss=False,
                l_use_model_loss=False,
                g_input_size=config.GLOBAL_IMAGE_SIZE,
                l_input_size=config.LOCAL_IMAGE_SIZE,
                threshold=0.50,
                verbose=config.VERBOSE):
    g_model, l_model, f_model = models
    g_criterion, l_criterion, f_criterion = criterions
    g_activation = get_hooks(g_model, params['GLOBAL_MODEL_NAME'])
    l_activation = get_hooks(l_model, params['LOCAL_MODEL_NAME'])

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
        for i, (inputs, labels, extra_features) in enumerate(loader):
            bz = inputs.shape[0]
            labels = labels.to(config.DEVICE)  # true labels
            extra_features = extra_features.to(config.DEVICE)
            y_true.append(labels)

            # Global branch
            g_inputs = F.resize(inputs, g_input_size)
            g_probs = g_model(g_inputs.to(config.DEVICE))  # pass through
            if g_use_model_loss:
                g_loss += g_model.loss(g_inputs, g_probs, labels, mean_error=True, reconstruct=False).item()
                g_probs = g_model.get_preduction(g_probs).to(config.DEVICE) 
            else:
                g_loss += float(torch.mean(g_criterion(g_probs, labels)))
                
            g_predicted = g_probs > threshold
            g_y_prob.append(g_probs)
            g_y_pred.append(g_predicted)

            # Local branch
            fm_global = g_activation['fm']  # get feature map
            local_patches = attention_gen_patches(inputs, fm_global, params['HEATMAP_THRESHOLD'])
            # TODO: can delegate resizing to attention_gen_patches
            l_inputs = F.resize(local_patches, l_input_size)
            l_probs = l_model(l_inputs).to(device)
            if l_use_model_loss:
                l_loss += l_model.loss(l_inputs, l_probs, labels, mean_error=True, reconstruct=False).item()
                l_probs = l_model.get_preduction(l_probs).to(config.DEVICE) 
            else:
                l_loss += float(torch.mean(l_criterion(l_probs, labels)))

            l_predicted = l_probs > threshold
            l_y_prob.append(l_probs)
            l_y_pred.append(l_predicted)

            # Fusion branch
            pool_g = g_activation['pool'].view(bz, -1)
            pool_l = l_activation['pool'].view(bz, -1)

            if not params['USE_EXTRA_INPUT']:
                extra_features = None
            f_probs = f_model(pool_g, pool_l, extra_features).to(device)
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
        print(f"Disease:{'':<22}AUROC")
        for i, lb in enumerate(config.TEXT_LABELS):
            print(f"{lb:<30}: {g_roc_aucs[i]:.4f}")
        print(f"\nROCAUC (Macro): {g_macro_roc_auc}")

        print("'" * 20)
        print("Local branch")
        print("'" * 20)
        print(f"Disease:{'':<22}AUROC")
        for i, lb in enumerate(config.TEXT_LABELS):
            print(f"{lb:<30}: {l_roc_aucs[i]:.4f}")
        print(f"\nAUROC (Macro): {l_macro_roc_auc}")

        print("'" * 20)
        print("Fusion branch")
        print("'" * 20)
        print(f"Disease:{'':<22}AUROC")
        for i, lb in enumerate(config.TEXT_LABELS):
            print(f"{lb:<30}: {f_roc_aucs[i]:.4f}")
        print(f"\nAUROC (Macro): {f_macro_roc_auc}")

    return [g_loss, l_loss, f_loss], \
           [g_macro_roc_auc, l_macro_roc_auc, f_macro_roc_auc], \
           [g_y_prob, l_y_prob, f_y_prob], \
           [g_y_pred, l_y_pred, f_y_pred], \
           y_true


def save_models(params, models, num_epochs, root_dir=config.ROOT_PATH, model_dir=config.MODEL_DIR, best=False):
    g_model, l_model, f_model = models

    model_subdir = 'model_' + config.WRITER_NAME.split('/')[-1]
    folder_path = os.path.join(root_dir, model_dir, model_subdir)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if best:
        g_path = os.path.join(folder_path, f'{params["GLOBAL_MODEL_NAME"]}_best.pth')
        l_path = os.path.join(folder_path, f'{params["LOCAL_MODEL_NAME"]}_best.pth')
        f_path = os.path.join(folder_path, f'{params["FUSION_MODEL_NAME"]}_best.pth')
    else:
        g_path = os.path.join(folder_path, f'{params["GLOBAL_MODEL_NAME"]}_{num_epochs}epoch.pth')
        l_path = os.path.join(folder_path, f'{params["LOCAL_MODEL_NAME"]}_{num_epochs}epoch.pth')
        f_path = os.path.join(folder_path, f'{params["FUSION_MODEL_NAME"]}_{num_epochs}epoch.pth')

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
        def hook(model, inp, output):
            act[name] = output.detach()
        return hook

    fm_name, pool_name = get_hook_names(model_name)

    # feature map and pooling of global model, pooling of local model
    activation['fm'] = operator.attrgetter(fm_name)(model).register_forward_hook(get_activation('fm', activation))
    activation['pool'] = operator.attrgetter(pool_name)(model).register_forward_hook(get_activation('pool', activation))
    return activation


def unfreeze_last_frozen_layer(model):
    last_frozen_layer = None
    # get the name of last frozen layer
    for name, child in reversed([_l for _l in model.named_children()]):  # layers of reversed order
        for p in child.parameters():
            if not p.requires_grad:
                last_frozen_layer = name
                break
        else:
            continue
        break
    # unfreeze
    if last_frozen_layer:
        for name, param in model.named_parameters():
            if name.startswith(last_frozen_layer):
                param.requires_grad = True
    # should be inplace, no need to return model
    return last_frozen_layer
