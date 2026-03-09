import re
import matplotlib.pyplot as plt

log_file = "outputs/log.txt"

steps = []
losses = []
loss_atts = []
loss_pres = []
accs = []
epochs = []
lrs = []

step_pat = re.compile(r"total step:\s*(\d+)", re.IGNORECASE)
epoch_pat = re.compile(r"epoch:\s*(\d+)", re.IGNORECASE)
lr_pat = re.compile(r"\(lr:\s*([0-9.eE+-]+)\)")
loss_pat = re.compile(r"\('loss',\s*([0-9.]+)\)")
loss_att_pat = re.compile(r"\('loss_att',\s*([0-9.]+)\)")
loss_pre_pat = re.compile(r"\('loss_pre',\s*([0-9.]+)\)")
acc_pat = re.compile(r"\('acc',\s*([0-9.]+)\)")

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "train, rank:" not in line:
            continue

        m_step = step_pat.search(line)
        m_epoch = epoch_pat.search(line)
        m_lr = lr_pat.search(line)
        m_loss = loss_pat.search(line)
        m_loss_att = loss_att_pat.search(line)
        m_loss_pre = loss_pre_pat.search(line)
        m_acc = acc_pat.search(line)

        if m_step and m_loss:
            steps.append(int(m_step.group(1)))
            losses.append(float(m_loss.group(1)))
            epochs.append(int(m_epoch.group(1)) if m_epoch else None)
            lrs.append(float(m_lr.group(1)) if m_lr else None)
            loss_atts.append(float(m_loss_att.group(1)) if m_loss_att else None)
            loss_pres.append(float(m_loss_pre.group(1)) if m_loss_pre else None)
            accs.append(float(m_acc.group(1)) if m_acc else None)

if not losses:
    print("No train loss found in outputs/log.txt")
else:
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses)
    plt.xlabel("Total step")
    plt.ylabel("Loss")
    plt.title("Paraformer Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print(f"Saved loss_curve.png with {len(losses)} points.")

    plt.figure(figsize=(10, 5))
    plt.plot(steps, loss_atts, label="loss_att")
    plt.plot(steps, loss_pres, label="loss_pre")
    plt.xlabel("Total step")
    plt.ylabel("Loss component")
    plt.title("Paraformer Loss Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_components.png", dpi=150)
    print("Saved loss_components.png")

    plt.figure(figsize=(10, 5))
    plt.plot(steps, accs)
    plt.xlabel("Total step")
    plt.ylabel("Acc")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("acc_curve.png", dpi=150)
    print("Saved acc_curve.png")

    plt.figure(figsize=(10, 5))
    plt.plot(steps, lrs)
    plt.xlabel("Total step")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lr_curve.png", dpi=150)
    print("Saved lr_curve.png")