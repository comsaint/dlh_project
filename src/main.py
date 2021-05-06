import numpy as np
import torch
from torch import nn
#from utils import writer
from train_model import train_model, eval_models
import config
from dataset import make_data_transform, load_data
from data_processing import load_data_file, make_train_test_split, make_train_val_split
from model import initialize_model, SimpleCLF
#from ray import tune


def main(params):
    print(f"Writer name: {config.WRITER_NAME}")
    # load labels
    df_data, lst_labels = load_data_file()
    print(f"Number of images: {len(df_data)}")
    
    # split the finding (disease) labels, to a list
    print(f"Number of labels: {len(lst_labels)}")
    print(f"Labels: {lst_labels}")

    # Map gender and view position to {0,1}
    gender_dict = {'M': 0, 'F': 1}
    view_dict = {'PA': 0, 'AP': 1}
    df_data.replace({"Patient Gender": gender_dict,
                     "View Position": view_dict},
                    inplace=True)

    # split data into train+val and test set
    df_train_val, df_test = make_train_test_split(df_data)
    # split train+val set into train and val set
    df_train, df_val = make_train_val_split(params, df_train_val)
    # make sure same patient does not exist in both train and val set
    assert set(df_train['Patient ID'].tolist()).isdisjoint(set(df_val['Patient ID'].tolist())), \
        "Same patient exist in train and validation set!"
    assert set(df_train['Patient ID'].tolist()).isdisjoint(set(df_test['Patient ID'].tolist())), \
        "Same patient exist in train and test set!"
    assert set(df_val['Patient ID'].tolist()).isdisjoint(set(df_test['Patient ID'].tolist())), \
        "Same patient exist in validation and test set!"

    # make sure all diseases appear at least 5 times in train and validation set
    if config.VERBOSE:
        print("Number of train images by disease:")
        print(df_train[config.TEXT_LABELS].sum())
        print("Number of validation images by disease:")
        print(df_val[config.TEXT_LABELS].sum())
    assert all(df_train[config.TEXT_LABELS].sum() >= 5), \
        "At least 1 disease appears less than 5 times in train set!"
    assert all(df_val[config.TEXT_LABELS].sum() >= 5), \
        "At least 1 disease appears less than 5 times in validation set!"

    print(f"Number of train images: {len(df_train)}")
    print(f"Number of val images: {len(df_val)}")
    print(f"Number of test images: {len(df_test)}")
    assert (len(df_train) + len(df_val) + len(df_test)) == len(df_data), \
        "Total number of images does not equal to sum of train+val+test!"

    # standardize the age
    age_mean = np.mean(df_train['Patient Age'])  # use training data to prevent leakage
    age_std = np.std(df_train['Patient Age'])
    df_train['Patient Age'] = (df_train['Patient Age'] - age_mean) / age_std
    df_val['Patient Age'] = (df_val['Patient Age'] - age_mean) / age_std
    df_test['Patient Age'] = (df_test['Patient Age'] - age_mean) / age_std

    # make data loaders
    tfx = make_data_transform(config.GLOBAL_IMAGE_SIZE, additional_transforms=params['AUGMENTATIONS'])
    train_data_loader = load_data(params, df_train, transform=tfx['train'], shuffle=True)
    val_data_loader = load_data(params, df_val, transform=tfx['test'], shuffle=False)
    test_data_loader = load_data(params, df_test, transform=tfx['test'], shuffle=False)

    # Criterion
    # Sigmoid + BCE loss https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # note we do the sigmoid here instead of inside model for numerical stability
    if params['USE_CLASS_WEIGHT']:
        class_weight = 1 / np.mean(df_data[lst_labels]) - 1  # ratio of #pos:#neg samples
        class_weight = torch.FloatTensor(class_weight.tolist()).to(config.DEVICE)
        g_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        l_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        f_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        print(f"Class weights:\n{class_weight}")
    else:
        g_criterion = nn.BCEWithLogitsLoss()
        l_criterion = nn.BCEWithLogitsLoss()
        f_criterion = nn.BCEWithLogitsLoss()

    criterions = [g_criterion, l_criterion, f_criterion]

    # train
    models, t_losses, v_losses, v_best_loss, v_rocs, roc_at_best_v_loss, best_model_paths = \
        train_model(
            params,
            train_loader=train_data_loader,
            val_loader=val_data_loader,
            criterions=criterions
            )
    print(f"Best Validation loss: {v_best_loss}; at which AUC = {roc_at_best_v_loss}")

    ##############################################
    # Test on the best models
    ##############################################
    print("Testing on the best model...")
    if config.VERBOSE:
        print(f"Best model paths: {best_model_paths}")
    g_best_path, l_best_path, f_best_path = best_model_paths
    g_model, g_input_size, _, g_fm_name, g_pool_name, g_fm_size = initialize_model(params, params['GLOBAL_MODEL_NAME'])
    l_model, l_input_size, _, _, l_pool_name, l_fm_size = initialize_model(params, params['LOCAL_MODEL_NAME'])
    if params['USE_EXTRA_INPUT']:
        s = g_fm_size + l_fm_size + 3
    else:
        s = g_fm_size + l_fm_size
    f_model = SimpleCLF(input_size=s).to(config.DEVICE)

    g_model.load_state_dict(torch.load(g_best_path))
    l_model.load_state_dict(torch.load(l_best_path))
    f_model.load_state_dict(torch.load(f_best_path))
    models = [g_model, l_model, f_model]

    test_loss, test_auc, _, _, _ = eval_models(params, models, test_data_loader, criterions)

    print(f"Test loss: {test_loss[2]}; Test ROC: {test_auc[2]}")
    '''
    writer.add_scalar("Loss/test", test_loss[2], 0)
    writer.add_scalar("ROCAUC/test", test_auc[2], 0)
    writer.flush()
    writer.close()
    '''
    print("End of main.")
    return {
        'g_auroc': test_auc[0],
        'l_auroc': test_auc[1],
        'f_auroc': test_auc[2]
        }


if __name__ == "__main__":
    '''
    if config.HYPERSEARCH:
        analysis = tune.run(
            main,
            config=config.params_hypersearch,
            resources_per_trial={"cpu": 4, "gpu": 1},
            name="exp_1",
            local_dir="./tune_results"
        )
        print("Best config: ", analysis.get_best_config(metric="f_auroc", mode="max"))
        # Get a dataframe for analyzing trial results.
        df = analysis.results_df
        df.to_csv('tune_result.csv')
        print("End search.")
    else:
        main(config.params)
    '''
    main(config.params)
