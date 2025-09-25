import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.manifold import TSNE
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.patches import Ellipse
import matplotlib as mpl


def load_model_and_tokenizer(model_name_or_path) -> Tuple[Accelerator, AutoModelForCausalLM, AutoTokenizer]:
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return accelerator, model, tokenizer


def plot_filled_ellipse(points, ax, color, scale=1.0, alpha=0.2):
    mean = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    if cov.shape != (2, 2):
        return
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * scale * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor=color, lw=1.5, alpha=alpha)
    ax.add_patch(ellipse)


def tsne_visualize(
        X, 
        y=None,
        title="t-SNE Visualization of High-Dimensional Data",
        xlabel="t-SNE Component 1",
        ylabel="t-SNE Component 2",
        save_path="tsne_visualization.png"
    ):
    mpl.rcParams['font.family'] = 'Times New Roman'
    if isinstance(X, torch.Tensor):
        X = X.detach().to(torch.float).cpu().numpy()

    sns.set(style="whitegrid", context="notebook", font_scale=1.2)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))

    if y is not None:
        y = np.array(y)
        label_map = {
            0: ("Aligned Knowledge", "#82a8e0"),
            1: ("Conflicting Knowledge", "#eb748b")
        }

        for label_value, (label_name, color) in label_map.items():
            indices = y == label_value
            class_points = X_tsne[indices]
            ax.scatter(
                class_points[:, 0],
                class_points[:, 1],
                c=color,
                label=label_name,
                s=50,
                alpha=0.9,
                edgecolors='none'
            )
            plot_filled_ellipse(class_points, ax, color=color, scale=1.5, alpha=0.25)

        ax.legend(fontsize=23, loc='lower right')
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c='#66c2a5', s=50, alpha=0.7, edgecolors='none')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return X_tsne


def attention_weights_visualize(attention_weights, tokens, output_path="attention_weights.png"):
    fig, ax = plt.subplots(figsize=(16, 12))
    if type(attention_weights) == torch.Tensor:
        attention_weights = attention_weights.detach().to(torch.float).cpu().numpy()
    cax = ax.matshow(attention_weights, cmap='viridis')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(attention_weights.shape[0]))

    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels([f'Head {i+1}' for i in range(attention_weights.shape[0])])

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.title("Attention Weights for the Last Token")
    plt.xlabel("Tokens")
    plt.ylabel("Attention Heads")
    plt.savefig(output_path)


def attention_matrix_visualize(attention_matrix, tokens, output_path="attention_matrix.png"):
    fig, ax = plt.subplots(figsize=(25, 25))
    if type(attention_matrix) == torch.Tensor:
        attention_matrix = attention_matrix.detach().to(torch.float).cpu().numpy()
    cax = ax.matshow(attention_matrix, cmap='viridis')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))

    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)


    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.title("Attention Matrix for the All the Tokens")
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.savefig(output_path)

