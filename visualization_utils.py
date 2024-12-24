from matplotlib import pyplot as plt   

def plot_semantic_differences(relationship, word1, word2):
    sd_names = list(relationship.keys())
    word1_projections = [relationship[sd][f'projection_{word1}'] for sd in sd_names]
    word2_projections = [relationship[sd][f'projection_{word2}'] for sd in sd_names]
    differences = [relationship[sd]['distance'] for sd in sd_names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the projections
    axes[0].bar(range(len(sd_names)), word1_projections, width=0.4, label=f"{word1}", align='center', alpha=0.7)
    axes[0].bar(range(len(sd_names)), word2_projections, width=0.4, label=f"{word2}", align='edge', alpha=0.7)
    axes[0].set_xticks(range(len(sd_names)))
    axes[0].set_xticklabels(sd_names, rotation=45)
    axes[0].set_ylabel("Projection Magnitude")
    axes[0].set_title("Projections of Words onto Semantic Dimensions")
    axes[0].legend()

    # Plot the distance (difference in projection)
    axes[1].bar(sd_names, differences, color='teal', alpha=0.7)
    axes[1].set_ylabel("Projection Difference")
    axes[1].set_title("Difference Between Words Across Semantic Dimensions")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
