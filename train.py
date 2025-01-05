import math
from dataloaders.dataset import ECGDataset
from models.model import ECGCodec
from torch import optim
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torchaudio
import argparse
import time
import json
import torch.nn.init as init

from benchmark import Result_Analysis

from models.losses import compute_loss_t, compute_loss_f


def ExponentialLR(optimizer, gamma: float = 0.999996):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


def Model_Init(net):
    for m in net.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)


# ----------------- model forward process ----------------------
def run_batch(batch, istrain: bool = False, use_disc: bool = False):
    output = {}

    inputs = batch["sig"].to(device)

    embedding_loss, recons, indices = model(inputs)

    output['loss_t'] = compute_loss_t(inputs, recons)
    output['loss_f'] = compute_loss_f(inputs, recons, multiscale_stft)
    output['loss_emb'] = embedding_loss

    output['loss'] = output['loss_t'] + output['loss_f'] + output['loss_emb']

    return output


# ----------------- training ----------------------
def train_epoch(use_disc: bool = False):
    global total_iter, training_epoch
    global ecg_model
    global ecg_loss_en

    model.train()

    total_loss = 0

    for i, batch in enumerate(train_loader):
        output = run_batch(batch, True, use_disc)

        loss = output['loss']
        total_loss += loss.item()

        # update encoder decoder
        optimizer.zero_grad()
        loss.backward()

        # gradient crop for training stability
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1e3
        )

        optimizer.step()
        scheduler.step()

        if (i + 1) % 10 == 0:
            print(
                "\tIter [%d/%d] Loss: %.4f"
                % (i + 1, len(train_loader), loss.item())
                , end=' '
            )
            loss_list = [f"{k}:{output[k]:.4f}" for k in output]
            loss_info = '\t' + '\t'.join(loss_list)
            print(loss_info)

        writer.add_scalar("Train loss (iter)", loss, total_iter)
        if use_disc:
            writer.add_scalar("Discriminator loss (iter)", output['loss_disc'], total_iter)
        total_iter += 1

    total_loss /= len(train_loader)
    print("Train loss - {:6f}".format(total_loss))
    writer.add_scalar("Train loss (epochs)", total_loss, training_epoch)

# ----------------- validation ----------------------
def val():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            output = run_batch(batch)
            loss = output['loss']
            total_loss += loss.item()

    total_loss /= len(val_loader)

    print("Validation loss - {:4f}".format(total_loss))
    writer.add_scalar("Validation loss", total_loss, training_epoch)
    return total_loss

# ----------------- test ----------------------
def test():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            output = run_batch(batch)
            loss = output['loss']
            total_loss += loss.item()

    total_loss /= len(test_loader)
    print("Test loss - {:4f}".format(total_loss))
    writer.add_scalar("Test loss", total_loss, training_epoch)

    return total_loss


# ------------------- loop ---------------------------
def loop():
    global training_epoch
    global best_epoch_loss
    global best_epoch
    global train_param
    global rst_dir

    for epoch in range(0, epochs):
        print("Epoch - {} LR - {}".format(training_epoch + 1, optimizer.state_dict()['param_groups'][0]['lr']))

        train_epoch()

        val_loss = val()

        if (epoch+1) % 10 == 0:
            test_loss = test()
            train_param["test_loss"] = test_loss
            train_param["epoch"] = epoch
            Result_Analysis(model, train_param, rst_dir)

        # save_checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), osp.join(pth_dir, "{:0>8}.pth".format(epoch)))

        if epoch == 0:
            best_epoch = epoch+1
            best_epoch_loss = val_loss
        else:
            if val_loss < best_epoch_loss:
                best_epoch = epoch + 1
                best_epoch_loss = val_loss

        training_epoch += 1

    # save best epoch info
    filename = osp.join(rst_dir, 'exp_result.csv')
    with open(filename, "a", encoding="utf-8") as f:
        best_info = f"best epoch, {best_epoch}, , best epoch loss, {best_epoch_loss},\n\n"
        f.write(best_info)
        f.close()


# train loop
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="./train_config/configs.json")
    parser.add_argument("--exp_time", default='temp')
    parser.add_argument("--encoder", default='seanet')
    parser.add_argument("--quantizer", default='vq')
    parser.add_argument("--bs", default=16)
    parser.add_argument("--lr", default=3e-4)
    parser.add_argument("--e_dims", default=4)
    parser.add_argument("--codebook_dims", default=16)
    parser.add_argument("--use_lookup", default="True")
    parser.add_argument("--bins", default=1024)
    parser.add_argument("--n_q", default=4)
    parser.add_argument("--ratios", default='[8,5,4,2]')
    parser.add_argument("--lr_scheduler", default="True")

    args = parser.parse_args()

    f = open(args.configs, 'r')
    content = f.read()
    configs = json.loads(content)
    f.close()

    # experiment information
    exp_name = configs["exp_name"]
    exp_dir = configs["exp_dir"]

    # model information
    encoder_name = configs["encoder"]
    quantizer = configs["quantizer"]

    # training parameters
    epochs = int(configs["epochs"])
    optimizer = configs["optim"]
    train_dataset = configs["train_dataset"]
    device = configs["device"]

    # dataset path
    train_json_path = configs["train_json_path"]
    val_json_path = configs["val_json_path"]
    test_json_path = configs["test_json_path"]

    # experiment adjustable parameters
    lr = float(args.lr)
    batch_size = int(args.bs)
    e_dims = int(args.e_dims)
    bins = int(args.bins)
    n_q = int(args.n_q)
    codebook_dims = int(args.codebook_dims)
    use_lookup = eval(args.use_lookup)

    lr_scheduler_en = eval(args.lr_scheduler)
    ecg_loss_en = eval(args.ecg_loss)
    encoder_pretrained = eval(args.pretrained)

    ratios = eval(args.ratios)
    hop_length = 1
    for item in ratios:
        hop_length *= item

    CR = hop_length * 11 / (n_q * math.log2(bins))
    # CR = hop_length * 11 / (32 * e_dims)

    # exp_info = f"bs_{str(batch_size)}_dims{e_dims}_bins{bins}_nq{n_q}"

    if use_lookup:
        exp_name = exp_name + "-lookup"
        exp_info = f"dims{e_dims}_cbdims{codebook_dims}_bins{bins}_nq{n_q}"
    else:
        exp_info = f"dims{e_dims}_bins{bins}_nq{n_q}"

    if args.exp_time == "temp":
        exp_time = str(time.asctime().replace(':', '_'))
        exp_time = exp_time.replace(' ', '-')
    else:
        exp_time = args.exp_time

    log_dir = osp.join(exp_dir, exp_name, exp_time, exp_info, "logs")
    pth_dir = osp.join(exp_dir, exp_name, exp_time, exp_info, "checkpoints")
    rst_dir = osp.join(exp_dir, exp_name, exp_time, exp_info, "result")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pth_dir, exist_ok=True)
    os.makedirs(rst_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # save training paramerters infomation
    train_param = {
        "dataset": "mit-bih",
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "optimizer": optimizer,
        "e_dims": e_dims,
        "codebook_dims": codebook_dims,
        "codebook_size": bins,
        "CR": CR,
        "n_q": n_q,
        "hop_length":hop_length,
    }
    param_filename = osp.join(exp_dir, exp_name, exp_time, 'train_param.json')
    with open(param_filename, 'a') as f:
        f.write(json.dumps(train_param))
    f.close()

    # print experiment info
    print("--------- Experiment Parameters ------------")
    print(f"Batch_size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning_Rate: {lr}")
    print(f"Embedding dims: {e_dims}")
    print(f"Codebook Size: {bins}")
    print(f"Compression Rate: {CR}")
    print(f"LR_scheduler: {lr_scheduler_en}")
    print(f"ECG Loss: {ecg_loss_en}")

    sample_rate = 360  # Unit: Hz
    segment = 60
    channels = 1

    model = ECGCodec.get_exp_model(ratios=ratios,
                                   e_dims=e_dims,
                                   codebook_dims=codebook_dims,
                                   codebook_size=bins,
                                   n_q=n_q,
                                   use_lookup=use_lookup)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer)

    # training metrics
    best_epoch = 1
    best_epoch_loss = 0

    # duration monitoring
    training_epoch = 0
    total_iter = 0

    # training data
    train_loader = ECGDataset(train_json_path).get_dataloader(batch_size=batch_size, num_workers=1, shuffle=True)
    val_loader = ECGDataset(val_json_path).get_dataloader(batch_size=batch_size, num_workers=1)
    # test data
    test_loader = ECGDataset(test_json_path).get_dataloader(batch_size=batch_size, num_workers=1)

    # stft transform
    scales = [7, 8, 9, 10, 11]
    multiscale_stft = []

    for item in scales:
        n_fft = 2 ** item
        win_length = n_fft
        hop_length = int(n_fft / 2)
        stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, window_fn=torch.hann_window,
            normalized=True, center=False, pad_mode=None, power=None)
        stft = stft.to(device)
        multiscale_stft.append(stft)

    loop()
