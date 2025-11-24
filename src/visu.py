import torch
import matplotlib.pyplot as plt


# -----------------------------
# 5. Visualization function
# -----------------------------
def visualize_feature_maps(model, loader, device, epoch, channels_per_layer):
    """
    - Take the first batch from loader
    - Forward through conv layers to get f1, f2, f3
    - For a single image (index 0), plot selected channels per layer
      as 2D feature maps.

    channels_per_layer: list of 3 lists:
        [
          [c1, c2, ...] for conv1,
          [c3, c4, ...] for conv2,
          [c5, c6, ...] for conv3
        ]
    """
    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        x = x.to(device)
        f1, f2, f3 = model.forward_feature_maps(x)

    # Take the first image in the batch
    fmaps = [f1[0].cpu(), f2[0].cpu(), f3[0].cpu()]  # each: (C,H,W)
    layer_names = ["conv1", "conv2", "conv3"]

    num_layers = 3
    max_cols = max(len(chs) for chs in channels_per_layer)

    fig, axes = plt.subplots(
        nrows=num_layers,
        ncols=max_cols,
        figsize=(3 * max_cols, 3 * num_layers)
    )

    if num_layers == 1:
        axes = [axes]  # normalize shape

    for row in range(num_layers):
        fmap = fmaps[row]  # (C,H,W)
        C, H, W = fmap.shape
        cols = channels_per_layer[row]

        for col_idx in range(max_cols):
            ax = axes[row][col_idx] if max_cols > 1 else axes[row]
            ax.axis("off")

            if col_idx < len(cols):
                ch = cols[col_idx]
                if ch < 0 or ch >= C:
                    ax.set_title(f"Invalid ch {ch}")
                    continue
                img = fmap[ch].numpy()
                ax.imshow(img, cmap="viridis")
                ax.set_title(f"{layer_names[row]} - ch {ch}")
            else:
                # no channel to show here
                ax.axis("off")

    plt.suptitle(f"Feature maps (epoch {epoch})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"fig_feature_maps/feature_maps_epoch_{epoch:03d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved feature map visualization for epoch {epoch}: {out_path}")
