import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

from utils import load_train_dataset, WrappedDataset, get_transform
from base_train import evaluate

# Dummy class for torch.load compatibility
class TaskDataset: pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === FGSM Attack ===
def fgsm_attack(model, x, y, loss_fn, epsilon=8/255):
    x_adv = x.clone().detach().requires_grad_(True)
    out = model(x_adv)
    loss = loss_fn(out, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# === PGD Attack (with curriculum steps) ===
def pgd_attack(model, x, y, loss_fn, epsilon, alpha, steps):
    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = torch.clamp(x_adv, 0, 1)
    for _ in range(steps):
        x_adv.requires_grad_()
        out = model(x_adv)
        loss = loss_fn(out, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# === EMA (Exponential Moving Average) ===
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {}
        self.decay = decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.detach()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

# === Main Training Function ===
def train(model, loader, val_loader, criterion, optimizer, scheduler, total_epochs):
    ema = EMA(model)

    for epoch in range(total_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        # === Curriculum Schedule ===
        if epoch < 7:
            # FGSM warmup
            epsilon, alpha, steps = 8/255, 0, 1  # alpha=0 disables PGD loop
            attack_fn = fgsm_attack
        elif epoch < 20:
            epsilon, alpha, steps = 2/255, 1/255, 1
            attack_fn = pgd_attack
        elif epoch < 40:
            epsilon, alpha, steps = 4/255, 1/255, 2
            attack_fn = pgd_attack
        else:
            epsilon, alpha, steps = 6/255, 2/255, 3
            attack_fn = pgd_attack

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if attack_fn == fgsm_attack:
                x_adv = fgsm_attack(model, x, y, criterion, epsilon)
            else:
                x_adv = pgd_attack(model, x, y, criterion, epsilon, alpha, steps)

            # Forward
            optimizer.zero_grad()
            logits_clean = model(x)
            logits_adv = model(x_adv)
            loss = 0.5 * criterion(logits_clean, y) + 0.5 * criterion(logits_adv, y)
            loss.backward()
            optimizer.step()

            ema.update(model)

            # Stats
            total_loss += loss.item()
            preds = torch.cat([logits_clean, logits_adv]).argmax(dim=1)
            targets = torch.cat([y, y])
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        acc = 100. * correct / total
        print(f"[Epoch {epoch+1:02d}] PGD-{steps} eps={epsilon:.4f} Loss={total_loss:.2f} Acc={acc:.2f}% LR={scheduler.get_last_lr()[0]:.6f}")

        # Evaluate EMA weights on val set
        ema.apply_shadow(model)
        clean_acc = evaluate(model, val_loader)
        ema.restore(model)
        print(f"\t EMA Clean Accuracy: {clean_acc:.2f}%")

    # === Clean Finetuning ===
    print("\n--- Final Finetuning on Clean Data ---")
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4

    for epoch in range(10):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            ema.update(model)

        ema.apply_shadow(model)
        acc = evaluate(model, val_loader)
        ema.restore(model)
        print(f"[Finetune {epoch+1}]  EMA Clean Accuracy: {acc:.2f}%")

    # Finalize EMA weights
    ema.apply_shadow(model)
    return model

# === Main Entry ===
if __name__ == "__main__":
    # Load and split dataset
    imgs, labels = load_train_dataset("Train.pt")
    dataset = WrappedDataset(imgs, labels, get_transform())
    n_train = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [n_train, len(dataset) - n_train], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

    # Define model
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    # Train full pipeline
    model = train(model, train_loader, val_loader, criterion, optimizer, scheduler, total_epochs=55)

    # Save
    torch.save(model.state_dict(), "fgsm_pgd_curriculum_ema.pt")
    print(" Saved model to fgsm_pgd_curriculum_ema.pt")
