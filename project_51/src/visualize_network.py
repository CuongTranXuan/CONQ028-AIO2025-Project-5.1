#!/usr/bin/env python3
"""
Store Transfer Network Matplotlib Visualizer

This script creates a matplotlib visualization of the store transfer network
as a complement to the GraphML export. It generates PNG images showing
the network layout with node sizes and edge weights.

Author: Generated for inventory optimization project
Date: October 2025
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def load_graphml_and_visualize(graphml_path: str, output_dir: str) -> None:
    """
    Load GraphML file and create matplotlib visualizations.

    Args:
        graphml_path (str): Path to the GraphML file
        output_dir (str): Directory to save visualization images
    """
    # Load the graph
    G = nx.read_graphml(graphml_path)

    # Convert string attributes back to numeric for calculations
    for node in G.nodes():
        G.nodes[node]["total_units"] = int(G.nodes[node]["total_units"])
        G.nodes[node]["total_units_in"] = int(G.nodes[node]["total_units_in"])
        G.nodes[node]["total_units_out"] = int(G.nodes[node]["total_units_out"])

    for u, v in G.edges():
        G.edges[u, v]["units_transferred"] = int(G.edges[u, v]["units_transferred"])
        G.edges[u, v]["weight"] = int(G.edges[u, v]["weight"])

    print(
        f"Loaded network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    # Create visualizations
    create_network_overview(G, output_dir)
    create_hub_analysis(G, output_dir)
    create_flow_intensity_map(G, output_dir)

    print(f"All visualizations saved to: {output_dir}")


def create_network_overview(G: nx.DiGraph, output_dir: str) -> None:
    """Create a comprehensive network overview visualization."""

    # Set up the figure
    plt.figure(figsize=(20, 16))

    # Use spring layout for better node distribution
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

    # Calculate node sizes based on total transfer volume
    node_sizes = [G.nodes[node]["total_units"] for node in G.nodes()]
    max_size = max(node_sizes) if node_sizes else 1
    min_size = min(node_sizes) if node_sizes else 1

    # Normalize node sizes (300 to 3000 for visibility)
    normalized_sizes = [
        (
            300 + 2700 * (size - min_size) / (max_size - min_size)
            if max_size > min_size
            else 1000
        )
        for size in node_sizes
    ]

    # Calculate edge weights for thickness
    edge_weights = [G.edges[u, v]["units_transferred"] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 1

    # Normalize edge widths (0.5 to 5.0)
    normalized_widths = [
        (
            0.5 + 4.5 * (weight - min_weight) / (max_weight - min_weight)
            if max_weight > min_weight
            else 2.0
        )
        for weight in edge_weights
    ]

    # Create color map for nodes based on net flow (in - out)
    net_flows = [
        G.nodes[node]["total_units_in"] - G.nodes[node]["total_units_out"]
        for node in G.nodes()
    ]
    node_colors = [
        "lightcoral" if flow < 0 else "lightblue" if flow > 0 else "lightgray"
        for flow in net_flows
    ]

    # Draw the network
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=normalized_sizes,
        node_color=node_colors,
        alpha=0.8,
        edgecolors="black",
        linewidths=1,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=normalized_widths,
        alpha=0.6,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        min_target_margin=15,
    )

    # Add labels with better positioning
    labels = {
        node: node.replace(" - ", "\\n").replace(" ", "\\n") for node in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")

    # Add title and legend
    plt.title(
        "Store Transfer Network Overview\\n"
        "Node size = Total transfer volume | Node color = Net flow direction\\n"
        "Edge thickness = Units transferred between stores",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Create legend
    red_patch = mpatches.Patch(
        color="lightcoral", label="Net Outflow (Supplier stores)"
    )
    blue_patch = mpatches.Patch(color="lightblue", label="Net Inflow (Receiver stores)")
    gray_patch = mpatches.Patch(color="lightgray", label="Balanced Flow")
    plt.legend(
        handles=[red_patch, blue_patch, gray_patch], loc="upper left", fontsize=12
    )

    # Add network statistics text box
    stats_text = (
        f"Network Statistics:\\n"
        f"Stores: {G.number_of_nodes()}\\n"
        f"Transfers: {G.number_of_edges()}\\n"
        f"Density: {nx.density(G):.3f}\\n"
        f"Avg Units/Transfer: {np.mean(edge_weights):.1f}"
    )

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.axis("off")
    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "network_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Network overview saved: {output_path.name}")


def create_hub_analysis(G: nx.DiGraph, output_dir: str) -> None:
    """Create visualization focusing on hub stores (highest degree nodes)."""

    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)

    # Get top 10 hub stores
    top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    hub_names = [hub[0] for hub in top_hubs]

    # Create subgraph with top hubs and their immediate neighbors
    hub_subgraph = G.subgraph(hub_names).copy()

    # Add neighbors of top hubs
    for hub in hub_names:
        neighbors = list(G.neighbors(hub)) + list(G.predecessors(hub))
        for neighbor in neighbors:
            if neighbor not in hub_subgraph:
                hub_subgraph.add_node(neighbor, **G.nodes[neighbor])

        # Add edges
        for neighbor in neighbors:
            if G.has_edge(hub, neighbor):
                hub_subgraph.add_edge(hub, neighbor, **G.edges[hub, neighbor])
            if G.has_edge(neighbor, hub):
                hub_subgraph.add_edge(neighbor, hub, **G.edges[neighbor, hub])

    plt.figure(figsize=(18, 14))

    # Position nodes with hubs at the center
    pos = nx.spring_layout(hub_subgraph, k=2, iterations=50, seed=42)

    # Distinguish hub nodes from others
    node_colors = [
        "gold" if node in hub_names else "lightsteelblue"
        for node in hub_subgraph.nodes()
    ]
    node_sizes = [2000 if node in hub_names else 1000 for node in hub_subgraph.nodes()]

    # Draw network
    nx.draw_networkx_nodes(
        hub_subgraph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        edgecolors="black",
        linewidths=2,
    )

    # Draw edges with varying thickness
    edge_weights = [
        hub_subgraph.edges[u, v]["units_transferred"] for u, v in hub_subgraph.edges()
    ]
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 1
    edge_widths = [
        (
            0.5 + 4 * (w - min_weight) / (max_weight - min_weight)
            if max_weight > min_weight
            else 2
        )
        for w in edge_weights
    ]

    nx.draw_networkx_edges(
        hub_subgraph,
        pos,
        width=edge_widths,
        alpha=0.6,
        edge_color="darkgray",
        arrows=True,
        arrowsize=15,
        arrowstyle="->",
    )

    # Add labels
    labels = {
        node: node.replace(" - ", "\\n").replace(" ", "\\n")
        for node in hub_subgraph.nodes()
    }
    nx.draw_networkx_labels(hub_subgraph, pos, labels, font_size=9, font_weight="bold")

    plt.title(
        "Hub Store Analysis - Top 10 Most Connected Stores\\n"
        "Gold nodes = Hub stores | Blue nodes = Connected stores",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add hub ranking text
    hub_text = "Top Hub Stores (by connections):\\n" + "\\n".join(
        [f"{i+1}. {hub[0]}" for i, hub in enumerate(top_hubs)]
    )
    plt.text(
        0.02,
        0.98,
        hub_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    plt.axis("off")
    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "hub_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Hub analysis saved: {output_path.name}")


def create_flow_intensity_map(G: nx.DiGraph, output_dir: str) -> None:
    """Create a heatmap-style visualization showing flow intensity."""

    plt.figure(figsize=(16, 12))

    # Use circular layout for better flow visualization
    pos = nx.circular_layout(G)

    # Calculate node sizes based on total units
    node_sizes = [G.nodes[node]["total_units"] for node in G.nodes()]
    max_node_size = max(node_sizes)
    normalized_node_sizes = [500 + 2000 * (size / max_node_size) for size in node_sizes]

    # Create color map based on units transferred
    edge_weights = [G.edges[u, v]["units_transferred"] for u, v in G.edges()]
    max_edge_weight = max(edge_weights)

    # Normalize edge colors (using a color map)
    edge_colors = [
        plt.cm.Reds(0.3 + 0.7 * (weight / max_edge_weight)) for weight in edge_weights
    ]
    edge_widths = [0.5 + 6 * (weight / max_edge_weight) for weight in edge_weights]

    # Draw nodes with size based on total volume
    node_colors = [
        plt.cm.Blues(0.3 + 0.7 * (size / max_node_size)) for size in node_sizes
    ]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=normalized_node_sizes,
        node_color=node_colors,
        alpha=0.8,
        edgecolors="black",
        linewidths=1,
    )

    # Draw edges with intensity-based colors
    for i, (u, v) in enumerate(G.edges()):
        nx.draw_networkx_edges(
            G,
            pos,
            [(u, v)],
            width=edge_widths[i],
            edge_color=[edge_colors[i]],
            alpha=0.8,
            arrows=True,
            arrowsize=12,
            arrowstyle="->",
        )

    # Add labels
    labels = {
        node: node.split(" ")[0] + "\\n" + " ".join(node.split(" ")[1:])
        for node in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold")

    plt.title(
        "Transfer Flow Intensity Map\\n"
        "Node color/size = Total volume | Edge color/thickness = Transfer intensity",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add color bars
    sm_nodes = plt.cm.ScalarMappable(
        cmap=plt.cm.Blues,
        norm=plt.Normalize(vmin=min(node_sizes), vmax=max(node_sizes)),
    )
    sm_nodes.set_array([])
    cbar1 = plt.colorbar(sm_nodes, ax=plt.gca(), shrink=0.6, pad=0.1, aspect=20)
    cbar1.set_label("Total Store Volume (units)", fontsize=10)

    plt.axis("off")
    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "flow_intensity_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Flow intensity map saved: {output_path.name}")


def main():
    """Main function to create matplotlib visualizations."""
    print("Creating matplotlib visualizations of store transfer network...")
    print("=" * 60)

    # Define file paths
    project_root = Path(__file__).parent.parent
    graphml_path = project_root / "visualizations" / "store_transfer_network.graphml"
    output_dir = project_root / "visualizations"

    try:
        if not graphml_path.exists():
            print(f"GraphML file not found: {graphml_path}")
            print(
                "Please run create_transfer_network.py first to generate the GraphML file."
            )
            return 1

        load_graphml_and_visualize(str(graphml_path), str(output_dir))

        print(f"\nSUCCESS! Matplotlib visualizations created:")
        print(f"   network_overview.png - Complete network with node/edge sizing")
        print(f"   hub_analysis.png - Focus on most connected stores")
        print(f"   flow_intensity_map.png - Heat map of transfer intensities")
        print(
            f"\nThese PNG files can be viewed in any image viewer or included in reports."
        )

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
