#!/usr/bin/env python3
"""
Store Transfer Network Visualization Generator

This script reads the best_transfer_plan.csv file and generates a GraphML representation
of the store transfer network for visualization in tools like yEd, Gephi, or Cytoscape.

Author: Generated for inventory optimization project
Date: October 2025
"""

import os
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd


def load_transfer_data(csv_path: str) -> pd.DataFrame:
    """
    Load the transfer plan data from CSV file.

    Args:
        csv_path (str): Path to the best_transfer_plan.csv file

    Returns:
        pd.DataFrame: Loaded transfer data
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} transfer records from {csv_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Transfer plan file not found: {csv_path}")
    except Exception as e:
        raise Exception(f"Error loading transfer data: {str(e)}")


def aggregate_transfers(df: pd.DataFrame) -> tuple:
    """
    Aggregate transfer data by store pairs and calculate node statistics.

    Args:
        df (pd.DataFrame): Raw transfer data

    Returns:
        tuple: (edge_data, node_data) dictionaries
    """
    # Aggregate transfers between store pairs
    edge_aggregation = {}

    # Node statistics
    node_stats = {}

    for _, row in df.iterrows():
        from_store = row["from_store_name"]
        to_store = row["to_store_name"]
        product = row["product_name"]
        units = row["units"]
        cost = row["transport_cost"]
        distance = row["distance_km"]

        # Edge aggregation
        edge_key = (from_store, to_store)
        if edge_key not in edge_aggregation:
            edge_aggregation[edge_key] = {
                "total_units": 0,
                "products": set(),
                "total_cost": 0.0,
                "avg_distance": 0.0,
                "distance_count": 0,
            }

        edge_aggregation[edge_key]["total_units"] += units
        edge_aggregation[edge_key]["products"].add(product)
        edge_aggregation[edge_key]["total_cost"] += cost
        edge_aggregation[edge_key]["avg_distance"] += distance
        edge_aggregation[edge_key]["distance_count"] += 1

        # Node statistics - from store (outgoing)
        if from_store not in node_stats:
            node_stats[from_store] = {
                "store_name": "",
                "store_id": None,
                "units_out": 0,
                "units_in": 0,
                "products_sent": set(),
                "products_received": set(),
            }

        node_stats[from_store]["store_name"] = from_store
        node_stats[from_store]["store_id"] = row["from_store_id"]
        node_stats[from_store]["units_out"] += units
        node_stats[from_store]["products_sent"].add(product)

        # Node statistics - to store (incoming)
        if to_store not in node_stats:
            node_stats[to_store] = {
                "store_name": "",
                "store_id": None,
                "units_out": 0,
                "units_in": 0,
                "products_sent": set(),
                "products_received": set(),
            }

        node_stats[to_store]["store_name"] = to_store
        node_stats[to_store]["store_id"] = row["to_store_id"]
        node_stats[to_store]["units_in"] += units
        node_stats[to_store]["products_received"].add(product)

    # Calculate average distances
    for edge_data in edge_aggregation.values():
        if edge_data["distance_count"] > 0:
            edge_data["avg_distance"] = (
                edge_data["avg_distance"] / edge_data["distance_count"]
            )

    print(f"Aggregated {len(edge_aggregation)} unique store-to-store transfers")
    print(f"Found {len(node_stats)} unique stores")

    return dict(edge_aggregation), dict(node_stats)


def create_network_graph(edge_data: dict, node_data: dict) -> nx.DiGraph:
    """
    Create a NetworkX directed graph from aggregated transfer data.

    Args:
        edge_data (dict): Aggregated edge data
        node_data (dict): Node statistics

    Returns:
        nx.DiGraph: NetworkX directed graph
    """
    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for store_name, stats in node_data.items():
        G.add_node(
            store_name,
            store_name=stats["store_name"],
            store_id=int(stats["store_id"]) if stats["store_id"] is not None else 0,
            total_units_in=stats["units_in"],
            total_units_out=stats["units_out"],
            total_units=stats["units_in"] + stats["units_out"],
            products_sent_count=len(stats["products_sent"]),
            products_received_count=len(stats["products_received"]),
            products_sent_list=", ".join(sorted(stats["products_sent"])),
            products_received_list=", ".join(sorted(stats["products_received"])),
        )

    # Add edges with attributes
    for (from_store, to_store), stats in edge_data.items():
        G.add_edge(
            from_store,
            to_store,
            units_transferred=stats["total_units"],
            number_of_products=len(stats["products"]),
            product_list=", ".join(sorted(stats["products"])),
            total_transport_cost=round(stats["total_cost"], 2),
            avg_distance_km=round(stats["avg_distance"], 2),
            weight=stats["total_units"],  # For visualization weight
        )

    print(
        f"Created network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    return G


def export_graphml(graph: nx.DiGraph, output_path: str) -> None:
    """
    Export the network graph to GraphML format.

    Args:
        graph (nx.DiGraph): NetworkX graph to export
        output_path (str): Path where to save the GraphML file
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write GraphML file
        nx.write_graphml(graph, output_path, encoding="utf-8", prettyprint=True)
        print(f"GraphML file exported successfully to: {output_path}")

        # Print some basic network statistics
        print(f"\nNetwork Statistics:")
        print(f"   - Nodes (stores): {graph.number_of_nodes()}")
        print(f"   - Edges (transfers): {graph.number_of_edges()}")
        print(f"   - Network density: {nx.density(graph):.3f}")

        # Top stores by total transfer volume
        node_volumes = [
            (node, data["total_units"]) for node, data in graph.nodes(data=True)
        ]
        node_volumes.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 5 Stores by Total Transfer Volume:")
        for i, (store, volume) in enumerate(node_volumes[:5], 1):
            print(f"   {i}. {store}: {volume:,} units")

        # Top transfers by units
        edge_volumes = [
            (f"{u} → {v}", data["units_transferred"])
            for u, v, data in graph.edges(data=True)
        ]
        edge_volumes.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 5 Store-to-Store Transfers:")
        for i, (transfer, units) in enumerate(edge_volumes[:5], 1):
            print(f"   {i}. {transfer}: {units:,} units")

    except Exception as e:
        raise Exception(f"Error exporting GraphML: {str(e)}")


def generate_network_summary(graph: nx.DiGraph, output_dir: str) -> None:
    """
    Generate a text summary of the network analysis.

    Args:
        graph (nx.DiGraph): NetworkX graph
        output_dir (str): Directory to save the summary
    """
    summary_path = os.path.join(output_dir, "network_analysis_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("STORE TRANSFER NETWORK ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        # Basic statistics
        f.write(f"Network Overview:\n")
        f.write(f"- Total Stores: {graph.number_of_nodes()}\n")
        f.write(f"- Total Store-to-Store Transfers: {graph.number_of_edges()}\n")
        f.write(f"- Network Density: {nx.density(graph):.4f}\n")
        total_degree = sum(d for n, d in graph.degree())
        avg_degree = (
            total_degree / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        )
        f.write(f"- Average Degree: {avg_degree:.2f}\n\n")

        # Node analysis
        f.write("Store Analysis:\n")
        node_data = [(node, data) for node, data in graph.nodes(data=True)]
        node_data.sort(key=lambda x: x[1]["total_units"], reverse=True)

        f.write("Top 10 Stores by Total Transfer Volume:\n")
        for i, (store, data) in enumerate(node_data[:10], 1):
            f.write(
                f"{i:2d}. {store:<30} | Total: {data['total_units']:>6,} units "
                f"(In: {data['total_units_in']:>6,}, Out: {data['total_units_out']:>6,})\n"
            )

        f.write("\nStores by Outgoing Transfers:\n")
        outgoing_sorted = sorted(
            node_data, key=lambda x: x[1]["total_units_out"], reverse=True
        )
        for i, (store, data) in enumerate(outgoing_sorted[:10], 1):
            f.write(
                f"{i:2d}. {store:<30} | Outgoing: {data['total_units_out']:>6,} units\n"
            )

        f.write("\nStores by Incoming Transfers:\n")
        incoming_sorted = sorted(
            node_data, key=lambda x: x[1]["total_units_in"], reverse=True
        )
        for i, (store, data) in enumerate(incoming_sorted[:10], 1):
            f.write(
                f"{i:2d}. {store:<30} | Incoming: {data['total_units_in']:>6,} units\n"
            )

        # Edge analysis
        f.write("\nTransfer Analysis:\n")
        edge_data = [(f"{u} → {v}", data) for u, v, data in graph.edges(data=True)]
        edge_data.sort(key=lambda x: x[1]["units_transferred"], reverse=True)

        f.write("Top 15 Store-to-Store Transfers:\n")
        for i, (transfer, data) in enumerate(edge_data[:15], 1):
            f.write(
                f"{i:2d}. {transfer:<50} | {data['units_transferred']:>6,} units "
                f"({data['number_of_products']} products)\n"
            )

    print(f"Network analysis summary saved to: {summary_path}")


def main():
    """
    Main function to orchestrate the GraphML generation process.
    """
    print("Starting Store Transfer Network Visualization Generation...")
    print("=" * 60)

    # Define file paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "results" / "best_transfer_plan.csv"
    output_dir = project_root / "visualizations"
    graphml_path = output_dir / "store_transfer_network.graphml"

    try:
        # Step 1: Load transfer data
        print("\nLoading transfer data...")
        df = load_transfer_data(str(csv_path))

        # Step 2: Aggregate transfers and calculate statistics
        print("\nAggregating transfer data...")
        edge_data, node_data = aggregate_transfers(df)

        # Step 3: Create network graph
        print("\nCreating network graph...")
        graph = create_network_graph(edge_data, node_data)

        # Step 4: Export to GraphML
        print("\nExporting to GraphML format...")
        export_graphml(graph, str(graphml_path))

        # Step 5: Generate analysis summary
        print("\nGenerating network analysis summary...")
        generate_network_summary(graph, str(output_dir))

        print(f"\nSUCCESS! Network visualization files created in: {output_dir}")
        print(f"   GraphML file: {graphml_path.name}")
        print(f"   Analysis summary: network_analysis_summary.txt")
        print(f"\nYou can now open the GraphML file in visualization tools like:")
        print(f"   - yEd Graph Editor")
        print(f"   - Gephi")
        print(f"   - Cytoscape")
        print(f"   - NetworkX (Python)")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
