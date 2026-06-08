# train.py

import os
import random
import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import FFT_CASwin_Classifier

from dataset import (
    load_dataset,
    create_dataloader,
    train_transform,
    test_transform
)

from utils import (
    calculate_acr,
    print_classification_report,
    plot_confusion_matrix
)

# ==========================================================
# REPRODUCIBILITY
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = "D:/Classification Using Unsupervised Models/train_fruit"

NUM_CLASSES = 10

EPOCHS = 60

BATCH_SIZE = 16

LEARNING_RATE = 1e-4

NUM_FOLDS = 10

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SAVE_DIR = "checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================================
# TRAIN FUNCTION
# ==========================================================

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer):

    model.train()

    running_loss = 0

    y_true = []
    y_pred = []

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(1)

        y_true.extend(
            labels.cpu().numpy()
        )

        y_pred.extend(
            preds.cpu().numpy()
        )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    return (
        running_loss / len(loader),
        acc
    )

# ==========================================================
# VALIDATION FUNCTION
# ==========================================================

def validate(
        model,
        loader,
        criterion):

    model.eval()

    running_loss = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            preds = outputs.argmax(1)

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    acr = calculate_acr(
        y_true,
        y_pred
    )

    return (
        running_loss / len(loader),
        accuracy,
        acr,
        y_true,
        y_pred
    )

# ==========================================================
# MAIN TRAINING
# ==========================================================

def main():

    image_paths, labels, classes = load_dataset(
        DATASET_PATH
    )

    image_paths = np.array(image_paths)

    labels = np.array(labels)

    kfold = KFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    fold_results = []

    fold_acr_results = []

    for fold, (
            train_idx,
            val_idx
    ) in enumerate(
        kfold.split(image_paths)
    ):

        print(
            f"\n{'='*60}"
        )

        print(
            f"Fold {fold+1}/{NUM_FOLDS}"
        )

        print(
            f"{'='*60}"
        )

        train_images = image_paths[
            train_idx
        ]

        val_images = image_paths[
            val_idx
        ]

        train_labels = labels[
            train_idx
        ]

        val_labels = labels[
            val_idx
        ]

        train_loader = create_dataloader(
            train_images,
            train_labels,
            batch_size=BATCH_SIZE,
            shuffle=True,
            transform=train_transform
        )

        val_loader = create_dataloader(
            val_images,
            val_labels,
            batch_size=BATCH_SIZE,
            shuffle=False,
            transform=test_transform
        )

        model = FFT_CASwin_Classifier(
            num_classes=NUM_CLASSES
        ).to(DEVICE)

        # =====================================================
        # CLASS-WEIGHTED LOSS FOR IMBALANCED DATA
        # =====================================================

        classes_unique = np.unique(train_labels)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes_unique,
            y=train_labels
        )

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32
        ).to(DEVICE)

        print("\nClass Weights:")
        print(class_weights)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        optimizer = Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )

        best_acr = 0

        best_y_true = None
        best_y_pred = None

        for epoch in range(EPOCHS):

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )

            val_loss, val_acc, val_acr, y_true, y_pred = validate(
                model,
                val_loader,
                criterion
            )

            scheduler.step()

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Train Loss:{train_loss:.4f} "
                f"Train Acc:{train_acc:.4f} "
                f"Val Loss:{val_loss:.4f} "
                f"Val Acc:{val_acc:.4f} "
                f"Val ACR:{val_acr:.2f}%"
            )

            if val_acr > best_acr:

                best_acr = val_acr

                best_y_true = y_true
                best_y_pred = y_pred

                save_path = os.path.join(
                    SAVE_DIR,
                    f"best_fold_{fold+1}.pth"
                )

                torch.save(
                    model.state_dict(),
                    save_path
                )

        fold_results.append(best_acr)

        fold_acr_results.append(best_acr)

        print(
            f"\nBest Fold ACR {fold+1}: "
            f"{best_acr:.2f}%"
        )

        print("\nClassification Report")

        print_classification_report(
            best_y_true,
            best_y_pred,
            classes
        )

        plot_confusion_matrix(
            best_y_true,
            best_y_pred,
            classes,
            save_path=os.path.join(
                RESULTS_DIR,
                f"confusion_matrix_fold_{fold+1}.png"
            )
        )

    print("\n")

    print("="*60)

    print(
        f"Final Average Classification Rate (ACR): "
        f"{np.mean(fold_acr_results):.2f}%"
    )

    print(
        f"ACR Standard Deviation : "
        f"{np.std(fold_acr_results):.2f}"
    )

    print("="*60)

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()