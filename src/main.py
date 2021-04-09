from train_model import train_model, eval_model
import config
from dataset import make_data_transform, load_data
from data_processing import load_data_file, train_test_split, make_train_test_split
from model import initialize_model
from train_model import save_model
import torch
from torch import nn
from torch import optim
from utils import writer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # load labels
    df_data, lst_labels = load_data_file(sampling=config.SAMPLING)
    print(f"Number of images: {len(df_data)}")
    # split the finding (disease) labels, to a list
    print(f"Number of labels: {len(lst_labels)}")
    print(f"Labels: {lst_labels}")
    # split data into train, val and test set
    df_train_val, df_test = make_train_test_split(df_data)

    # use stratify, especially for imbalance dataset
    df_train, df_val = train_test_split(df_train_val)
    print(f"Number of train images: {len(df_train)}")
    print(f"Number of val images: {len(df_val)}")
    print(f"Number of test images: {len(df_test)}")
    assert (len(df_train) + len(df_val) + len(df_test)) == len(df_data), \
        "Total number of images does not equal to sum of train+val+test!"

    # Initialize the model for this run
    model, input_size = initialize_model(config.MODEL_NAME, config.NUM_CLASSES)
    model = model.to(device)
    print(f"Input image size: {input_size}")

    # make data loader
    tfx = make_data_transform(input_size)
    train_data_loader = load_data(df_train, transform=tfx['train'], shuffle=True)
    val_data_loader = load_data(df_val, transform=tfx['test'], shuffle=False)

    # visualize the model in TensorBoard
    _img, _ = next(iter(train_data_loader))
    writer.add_graph(model, _img.to(device))

    # Criterion
    # Sigmoid + BCE loss https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # note we do the sigmoid here instead of inside model for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    # to unfreeze more trainable layers, use: `optimizer.add_param_group({'params': model.<layer>.parameters()})`
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)

    # train
    model, t_losses, v_losses, v_best_loss, v_rocs, roc_at_best_v_loss, best_model_pth = \
        train_model(model, train_data_loader, val_data_loader, criterion, optimizer, num_epochs=config.NUM_EPOCHS,
                    verbose=False)
    # save the final model
    save_model(model, 999)

    # load and test on the best model
    model.load_state_dict(torch.load(best_model_pth))
    test_data_loader = load_data(df_test, transform=tfx['test'], shuffle=False)
    test_loss, test_auc, _, _, _ = eval_model(model.to(device), test_data_loader, criterion)
    print(f"Best Validation loss: {v_best_loss}; at which AUC = {roc_at_best_v_loss}")
    print(f"Test loss: {test_loss}; Test ROC: {test_auc}")
    writer.add_scalar("Loss/test", test_loss, 0)
    writer.add_scalar("ROCAUC/test", test_auc, 0)
    writer.flush()
    writer.close()
    print("End of script.")
    return None


if __name__ == "__main__":
    main()
