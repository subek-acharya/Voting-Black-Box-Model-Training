import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import utils
from ModelFactory import ModelFactory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CaiT or a linear SVM on the Voter grayscale dataset."
    )
    parser.add_argument(
        "--model",
        choices=["cait", "svm"],
        required=True,
        help="Which model to train.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (SGD).")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing the Voter .pth datasets.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save trained checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. 'cuda' or 'cpu'). Defaults to cuda if available.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(
    args,
    model,
    device,
    loss_fn,
    predict_correct_fn,
    label_transform,
    ckpt_name,
):
    """Shared training loop used by both CaiT and SVM.

    Args:
        loss_fn(output, labels) -> scalar loss tensor
        predict_correct_fn(output, labels) -> number of correct predictions (int)
        label_transform(labels) -> labels fed to loss/accuracy (e.g. {-1,+1} for SVM)
        ckpt_name: checkpoint filename under args.checkpoint_dir
    """
    trainLoader = utils.GetVoterTrainingCombined(args.batch_size, args.data_dir)
    valLoader = utils.GetVoterValidation(args.batch_size, args.data_dir)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch: {epoch}")
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(trainLoader)
        for batch_idx, (inputs, targets) in enumerate(trainLoader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            labels = label_transform(targets)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += predict_correct_fn(outputs, labels)

            utils.progress_bar(
                batch_idx,
                num_batches,
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )

        scheduler.step()

        # Validation + best-accuracy checkpointing.
        val_acc = evaluate(model, valLoader, device, predict_correct_fn, label_transform)
        print("Validation accuracy: %.3f%%" % (100.0 * val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"state_dict": model.state_dict(), "acc": val_acc, "epoch": epoch},
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path} (acc {100.0 * val_acc:.3f}%)")

    print(f"\nTraining complete. Best validation accuracy: {100.0 * best_acc:.3f}%")


def evaluate(model, valLoader, device, predict_correct_fn, label_transform):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in valLoader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            labels = label_transform(targets)
            outputs = model(inputs)
            total += targets.size(0)
            correct += predict_correct_fn(outputs, labels)
    return correct / float(total)


def train_cait(args, device):
    model = ModelFactory(device).get_model("cait")
    criterion = nn.CrossEntropyLoss()

    def loss_fn(outputs, labels):
        return criterion(outputs, labels)

    def predict_correct_fn(outputs, labels):
        preds = outputs.argmax(dim=1)
        return int((preds == labels).sum().item())

    def label_transform(targets):
        # CaiT uses the integer {0,1} class labels directly.
        return targets.long()

    run_training(
        args,
        model,
        device,
        loss_fn=loss_fn,
        predict_correct_fn=predict_correct_fn,
        label_transform=label_transform,
        ckpt_name="cait_voter.pth",
    )


def train_svm(args, device):
    model = ModelFactory(device).get_model("svm")

    def loss_fn(outputs, labels):
        return model.loss(outputs, labels)

    def predict_correct_fn(outputs, labels):
        # Sign-match accuracy; sign(0) counts as incorrect (edge case).
        return int((torch.sign(outputs) == labels).sum().item())

    def label_transform(targets):
        # Map {0,1} -> {-1,+1} for the hinge loss.
        return targets.float() * 2 - 1

    run_training(
        args,
        model,
        device,
        loss_fn=loss_fn,
        predict_correct_fn=predict_correct_fn,
        label_transform=label_transform,
        ckpt_name="svm_voter.pth",
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if args.model == "cait":
        train_cait(args, device)
    elif args.model == "svm":
        train_svm(args, device)


if __name__ == "__main__":
    main()
