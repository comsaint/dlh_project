from train_model import train_model, eval_model, save_model
import config
from dataset import make_data_transform, load_data
from data_processing import load_data_file, train_test_split, make_train_test_split
from model import initialize_model
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
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
    df_train, df_val = train_test_split(df_train_val, stratify_label=config.DISEASE)
    print(f"Number of train images: {len(df_train)}")
    print(f"Number of val images: {len(df_val)}")
    print(f"Number of test images: {len(df_test)}")
    assert (len(df_train) + len(df_val) + len(df_test)) == len(df_data), \
        "Total number of images does not equal to sum of train+val+test!"

    # Criterion, with class weight
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    num_neg = sum(df_train[config.DISEASE] == 0)
    num_pos = sum(df_train[config.DISEASE] == 1)
    assert num_neg + num_pos == len(df_train)
    print(f"# of negative/positive cases: {num_neg}:{num_pos}")
    class_weight = torch.FloatTensor([(1 / num_neg) * (len(df_train)) / 2.0, (1 / num_pos) * (len(df_train)) / 2.0]).to(
        device)
    print(f"Class weight: {class_weight}")
    criterion = nn.CrossEntropyLoss(weight=class_weight)  # note the class weight

    # Initialize the model for this run
    model, input_size = initialize_model(config.MODEL_NAME, config.NUM_CLASSES, config.FEATURE_EXTRACT, use_pretrained=True)
    model = model.to(device)
    print(model)  # print structure
    print(f"Input image size: {input_size}")

    # make data loader
    tfx = make_data_transform(input_size)
    train_data_loader = load_data(df_train, label=config.DISEASE, transform=tfx['train'], shuffle=True)
    val_data_loader = load_data(df_val, label=config.DISEASE, transform=tfx['test'], shuffle=False)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # train
    model, t_losses, v_losses, v_best_auc, v_rocs, best_model_pth = train_model(model, train_data_loader,
                                                                                val_data_loader, criterion, optimizer,
                                                                                num_epochs=config.NUM_EPOCHS,
                                                                                verbose=False)

    # load and test on the best model
    model.load_state_dict(torch.load(best_model_pth))
    test_data_loader = load_data(df_test, label=config.DISEASE, transform=tfx['test'], shuffle=False)
    test_loss, test_auc, _, _, _ = eval_model(model.to(device), test_data_loader, criterion)
    print(f"Test loss: {test_loss}; Test ROC: {test_auc}")
    print("End of script.")
    return None


if __name__ == "__main__":
    main()
    writer.close()
