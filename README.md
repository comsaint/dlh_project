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
8. After training completes, look at the result on console. Also, the trained models are under `model/` folder.
