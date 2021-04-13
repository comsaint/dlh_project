# How to run
1. Clone the repository.
2. Create and activate a virtual environment ,e.g. `python3 -m venv -n myenv`.
3. In root folder `dlh_project`, run the setup script.
    - Linux/Mac: `setup.sh`
    - Windows: `setup_win.sh`

    If you encounter errors installing packages due to different hardware/software,
    try install the following package via `pip` manually (and comment out the `pip install -r requirements.txt` line in setup script). You will need the following packages:
    - PyTorch, PyTorchVision and CudaTookit (for GPU): check [this link](https://pytorch.org/get-started/locally/) for command
    - Kaggle (check [here](https://www.kaggle.com/docs/api) for setup. **This is required regardless of installation method**)
    - Pillow (or Pillow-SIMD for Linux/Mac)
    - Scikit-Learn
    - Pandas
    - Matplotlib
    - TensorBoard

4. The script will download images and other data from Kaggle, then extract and organize the data in proper folder. This may take several hours.
5. After setup completes, `cd` into the `src` folder: `cd src`.
6. (Optional) open `config.py` and change settings, such as the model to tune, number of epochs etc.
7. Run `python main.py`. You will see the training progress in console print.
    - (Optional) you may monitor the training loss with TensorBoard by running `tensorboard --logdir=runs` in `\src`.
8. After training completes, look at the result on console. Also, the trained models are under `model/` folder.

# Notes

1. The `src/config.py` is the primary configuration file.
   - `NUM_CLASS` must be either set to **14** (for each disease) or **15** (14 diseases + 1 'No Finding').
   - `LEARNING_RATE` controls the training process. High learning rate leads to faster training, but may suffer from oscillation (loss moving up and down) or overshoot (training loss goes up). It will be controlled by a scheduler later.
   - Different `SEED` leads to different result.
2. You may use TensorBoard to monitor the training process: `cd` to `src` and run `tensorboard --logdir=runs`.
3. Trained models are under `./models`. By default the model is saved every 5 epochs, plus the final model and the model with the best validation loss are also saved. There is a script `src/test_model.py` which you can use to quickly test the performance of a saved model given its path and architecture.
