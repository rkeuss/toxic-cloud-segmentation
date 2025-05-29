import pandas as pd
import matplotlib.pyplot as plt

filenames = [
    'trainlosses_cross_entropy_directional.csv',
    'trainlosses_cross_entropy_hybrid.csv',
    'trainlosses_cross_entropy_local.csv',
    'trainlosses_cross_entropy_pixel.csv',
    'trainlosses_dice_directional.csv',
    'trainlosses_dice_hybrid.csv',
    'trainlosses_dice_local.csv',
    'trainlosses_dice_pixel.csv'
]
for filename in filenames:
    df = pd.read_csv(f'results/{filename}')
    df["total_loss"] = df["supervised_loss"] + 0.1 * df["semi_supervised_loss"]
    df["global_step"] = range(1, len(df) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(df["global_step"], df["total_loss"], marker='o')
    plt.xlabel("Training Step (Cumulative Epochs Across Folds)")
    plt.ylabel(f"Total Loss {filename}")
    plt.title("Total Loss Over Training")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    df_semi = df[df["semi_supervised_loss"] > 0].copy()
    df_semi["global_step"] = range(1, len(df_semi) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(df_semi["global_step"], df_semi["semi_supervised_loss"], marker='o')
    plt.xlabel("Training Step (Cumulative Epochs Across Folds)")
    plt.ylabel("Semi-Supervised Loss")
    plt.title(f"Semi-Supervised Loss Over Training {filename}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()