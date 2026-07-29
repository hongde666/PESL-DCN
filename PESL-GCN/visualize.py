import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dependency_adjacency_matrix(tokens, adj_matrix, save_path="adjacency_matrix.png"):
    """
    可视化句法依存邻接矩阵 A (Eq. 6)[cite: 1]
    """
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(
        adj_matrix[:len(tokens), :len(tokens)],
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar=True
    )
    plt.title("PESL-GCN Syntactic Dependency Adjacency Matrix", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Adjacency matrix heatmap successfully saved to {save_path}")

if __name__ == "__main__":
    # 示例调用
    sample_tokens = ["Phosphorylation", "of", "STAT3", "inhibits", "cell", "proliferation"]
    # 模拟生成的 6x6 邻接矩阵
    dummy_adj = np.eye(6) + np.random.uniform(0, 0.3, (6, 6))
    plot_dependency_adjacency_matrix(sample_tokens, dummy_adj)