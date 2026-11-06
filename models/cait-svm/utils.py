import os
import sys
import time
import shutil

import torch
from torch.utils.data import TensorDataset, DataLoader

# ------------ Voter dataset loaders -----------------
# Ported from the sibling Spiking-Neural-Network/utils.py. Behavior is preserved
# verbatim; the only change is that the hard-coded "./data/<file>" paths are made
# configurable through a `data_dir` argument (default "./data"). No downloads.


def GetVoterTrainingCombined(batchSize, data_dir="./data"):
    trainData = torch.load(
        os.path.join(data_dir, "kaleel_final_dataset_train_Combined_Grayscale.pth"),
        weights_only=False,
    )
    trainImages = trainData["data"].float()
    trainLabels = trainData["binary_labels"].long()
    trainDataset = TensorDataset(trainImages, trainLabels)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    return trainLoader


def GetVoterValidation(batchSize, data_dir="./data"):
    valData = torch.load(
        os.path.join(data_dir, "kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth"),
        weights_only=False,
    )
    valImages = valData["data"].float()
    valLabels = valData["binary_labels"].long()
    valDataset = TensorDataset(valImages, valLabels)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)
    return valLoader


# Validate using a dataloader (argmax match, as in the sibling repo).
def validateD(valLoader, model, device=None):
    # switch to evaluate mode
    model.eval()
    acc = 0
    batchTracker = 0
    with torch.no_grad():
        # Go through and process the data in batches
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0]  # Number of samples in this batch
            batchTracker = batchTracker + sampleSize
            if device is None:  # assume cuda
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            # compute output
            output = model(inputVar)
            output = output.float()
            # Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc + 1
    acc = acc / float(len(valLoader.dataset))
    return acc


# ######## CLI visualization functions #####################
# Get terminal width with fallback
try:
    term_width = shutil.get_terminal_size().columns
except Exception:
    term_width = 80  # Default fallback

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')
    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))
    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
