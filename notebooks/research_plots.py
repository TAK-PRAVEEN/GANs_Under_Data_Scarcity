import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE SETTINGS
# =========================

plt.rcParams.update({

    "font.family": "serif",
    "font.serif": ["Times New Roman"],

    "font.size": 12,

    "axes.labelsize": 12,
    "axes.titlesize": 13,

    "legend.fontsize": 10,

    "lines.linewidth": 2.5,

    "figure.figsize": (6,4),

    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4

})

# =========================
# PATH SETTINGS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.path.join(BASE_DIR, "experiments", "logs")

SAVE_DIR = os.path.join(BASE_DIR, "experiments", "plots")

os.makedirs(SAVE_DIR, exist_ok=True)

losses = ["vanilla", "lsgan", "wgan", "wgan_gp"]
splits = [1000, 5000, 10000, 25000]

colors = {

    "vanilla": "#1f77b4",   # blue
    "lsgan": "#2ca02c",     # green
    "wgan": "#ff7f0e",      # orange
    "wgan_gp": "#d62728"    # red

}


# =========================
# LOAD DATA
# =========================

def load_all_logs():

    all_logs = []

    for loss in losses:

        for split in splits:

            path = f"{LOG_DIR}/dcgan_{loss}_{split}.csv"

            if os.path.exists(path):

                df = pd.read_csv(path)

                df["loss_type"] = loss
                df["data_size"] = split

                all_logs.append(df)

    return pd.concat(all_logs)


logs = load_all_logs()


# =========================
# 1. GENERATOR LOSS PLOT
# =========================

for split in splits:

    plt.figure()

    for loss in losses:

        subset = logs[
            (logs.loss_type == loss) &
            (logs.data_size == split)
        ]

        plt.plot(
            subset.epoch,
            subset.g_loss,
            label=loss.upper(),
            color=colors[loss]
        )

    plt.xlabel("Epoch")
    plt.ylabel("Generator Loss")

    plt.title(f"Generator Loss (Dataset Size = {split})")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{SAVE_DIR}/generator_loss_{split}.svg",
        format="svg"
    )

    plt.savefig(
        f"{SAVE_DIR}/generator_loss_{split}.pdf",
        format="pdf"
    )

    plt.close()


# =========================
# 2. DISCRIMINATOR LOSS
# =========================

for split in splits:

    plt.figure()

    for loss in losses:

        subset = logs[
            (logs.loss_type == loss) &
            (logs.data_size == split)
        ]

        plt.plot(
            subset.epoch,
            subset.d_loss,
            label=loss.upper(),
            color=colors[loss]
        )

    plt.xlabel("Epoch")
    plt.ylabel("Discriminator Loss")

    plt.title(f"Discriminator Loss (Dataset Size = {split})")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{SAVE_DIR}/discriminator_loss_{split}.svg"
    )

    plt.savefig(
        f"{SAVE_DIR}/discriminator_loss_{split}.pdf"
    )

    plt.close()


# =========================
# 3. STABILITY PLOT
# =========================

stability = []

for loss in losses:

    for split in splits:

        subset = logs[
            (logs.loss_type == loss) &
            (logs.data_size == split)
        ]

        stability.append({

            "loss": loss,
            "split": split,
            "g_std": subset.g_loss.std()

        })


stab_df = pd.DataFrame(stability)


plt.figure()

for loss in losses:

    subset = stab_df[stab_df.loss == loss]

    plt.plot(
        subset.split,
        subset.g_std,
        marker="o",
        label=loss.upper(),
        color=colors[loss]
    )


plt.xlabel("Dataset Size")
plt.ylabel("Generator Loss Std Dev")

plt.title("Training Stability Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{SAVE_DIR}/stability_plot.svg"
)

plt.savefig(
    f"{SAVE_DIR}/stability_plot.pdf"
)

plt.close()



# =========================
# 4. TRAINING TIME PLOT
# =========================

time_df = logs.groupby(
    ["loss_type","data_size"]
)["epoch_time"].mean().reset_index()


plt.figure()

for loss in losses:

    subset = time_df[time_df.loss_type == loss]

    plt.plot(
        subset.data_size,
        subset.epoch_time,
        marker="o",
        label=loss.upper(),
        color=colors[loss]
    )


plt.xlabel("Dataset Size")
plt.ylabel("Epoch Time (seconds)")

plt.title("Training Cost Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{SAVE_DIR}/training_time.svg"
)

plt.savefig(
    f"{SAVE_DIR}/training_time.pdf"
)

plt.close()



print("All Research Plots Saved Successfully.")