import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SimplifiedResults:
    def __init__(self, output_dir="results"):
        """
        Initialize simplified results generator.

        Args:
            output_dir: Directory to save simplified results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set clean plot style
        plt.style.use("default")
        sns.set_palette("Set2")

        # Core colors for consistent visualization
        self.colors = {
            "excess": "#FF6B6B",  # Red for excess inventory
            "needed": "#4ECDC4",  # Teal for needed inventory
            "balanced": "#45B7D1",  # Blue for balanced
            "transfer": "#96CEB4",  # Green for transfers
            "cost": "#FECA57",  # Yellow for costs
        }

    def generate_executive_summary(
        self, analysis_df, transfer_plans, impact_data, stores_df, products_df
    ):
        """
        Generate a single executive summary with key metrics and recommendations.

        Args:
            analysis_df: Inventory analysis results
            transfer_plans: Dictionary of transfer plans by algorithm
            impact_data: Dictionary of impact results by algorithm
            stores_df: Store information
            products_df: Product information
        """
        print("\n=== GENERATING EXECUTIVE SUMMARY ===")

        # Create summary data
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_stores": len(stores_df),
            "total_products": len(products_df),
            "analysis_period": "Last 30 days",
        }

        # Current inventory status
        status_counts = analysis_df["inventory_status"].value_counts()
        total_items = len(analysis_df)

        summary.update(
            {
                "excess_items": status_counts.get("Excess", 0),
                "needed_items": status_counts.get("Needed", 0),
                "balanced_items": status_counts.get("Balanced", 0),
                "excess_percentage": round(
                    status_counts.get("Excess", 0) / total_items * 100, 1
                ),
                "imbalance_severity": self._calculate_imbalance_severity(analysis_df),
            }
        )

        # Best optimization results
        if transfer_plans and impact_data:
            best_algorithm, best_results = self._find_best_algorithm(
                transfer_plans, impact_data
            )
            summary.update(
                {
                    "recommended_algorithm": best_algorithm,
                    "recommended_transfers": len(best_results["plan"]),
                    "estimated_cost_savings": best_results["cost_savings"],
                    "inventory_improvement": best_results["inventory_improvement"],
                }
            )

        # Save executive summary
        self._save_executive_summary(summary)

        # Generate key visualizations
        self._create_key_visualizations(
            analysis_df, transfer_plans, impact_data, stores_df
        )

        # Generate actionable transfer recommendations
        if transfer_plans:
            self._generate_transfer_recommendations(
                transfer_plans, stores_df, products_df
            )

    def _calculate_imbalance_severity(self, analysis_df):
        """Calculate overall inventory imbalance severity."""
        excess_count = len(analysis_df[analysis_df["inventory_status"] == "Excess"])
        needed_count = len(analysis_df[analysis_df["inventory_status"] == "Needed"])
        total_count = len(analysis_df)

        imbalance_ratio = (excess_count + needed_count) / total_count

        if imbalance_ratio > 0.4:
            return "HIGH"
        elif imbalance_ratio > 0.2:
            return "MEDIUM"
        else:
            return "LOW"

    def _find_best_algorithm(self, transfer_plans, impact_data):
        """Find the best performing algorithm based on key metrics."""
        algorithm_scores = {}

        for algorithm in transfer_plans.keys():
            if algorithm in impact_data and impact_data[algorithm] is not None:
                impact = impact_data[algorithm]
                plan = transfer_plans[algorithm]

                # Calculate composite score based on:
                # 1. Cost efficiency (lower is better)
                # 2. Inventory improvement (higher is better)
                # 3. Number of transfers (fewer is better for simplicity)

                try:
                    cost = impact.get("Transfer Plan", {}).get(
                        "Total Transport Cost", 0
                    )
                    improvement = impact.get("Improvement", {}).get(
                        "Increase in Balanced Items", 0
                    )
                    num_transfers = len(plan) if not plan.empty else 0

                    # Normalize and combine scores (simple heuristic)
                    cost_score = 1 / (1 + cost / 1000)  # Lower cost is better
                    improvement_score = (
                        improvement / 100
                    )  # Higher improvement is better
                    simplicity_score = 1 / (
                        1 + num_transfers / 10
                    )  # Fewer transfers is better

                    composite_score = cost_score + improvement_score + simplicity_score

                    algorithm_scores[algorithm] = {
                        "score": composite_score,
                        "plan": plan,
                        "cost_savings": cost,
                        "inventory_improvement": improvement,
                    }
                except:
                    continue

        if algorithm_scores:
            best_algorithm = max(
                algorithm_scores.keys(), key=lambda k: algorithm_scores[k]["score"]
            )
            return best_algorithm, algorithm_scores[best_algorithm]

        return "Rule-Based", {
            "plan": pd.DataFrame(),
            "cost_savings": 0,
            "inventory_improvement": 0,
        }

    def _save_executive_summary(self, summary):
        """Save executive summary to a simple text file."""
        output_path = os.path.join(self.output_dir, "EXECUTIVE_SUMMARY.txt")

        with open(output_path, "w") as f:
            f.write("INVENTORY OPTIMIZATION EXECUTIVE SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Report Generated: {summary['timestamp']}\n")
            f.write(f"Analysis Period: {summary['analysis_period']}\n\n")

            f.write("CURRENT SITUATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Stores: {summary['total_stores']}\n")
            f.write(f"Total Products: {summary['total_products']}\n")
            f.write(
                f"Items with Excess Inventory: {summary['excess_items']} ({summary['excess_percentage']}%)\n"
            )
            f.write(f"Items Needing More Inventory: {summary['needed_items']}\n")
            f.write(f"Well-Balanced Items: {summary['balanced_items']}\n")
            f.write(f"Imbalance Severity: {summary['imbalance_severity']}\n\n")

            if "recommended_algorithm" in summary:
                f.write("RECOMMENDED ACTION:\n")
                f.write("-" * 20 + "\n")
                f.write(
                    f"Best Optimization Method: {summary['recommended_algorithm']}\n"
                )
                f.write(
                    f"Recommended Transfers: {summary['recommended_transfers']} transfers\n"
                )
                f.write(
                    f"Estimated Transport Cost: ${summary['estimated_cost_savings']:,.2f}\n"
                )
                f.write(
                    f"Expected Inventory Improvement: {summary['inventory_improvement']} items\n\n"
                )

            f.write("NEXT STEPS:\n")
            f.write("-" * 20 + "\n")
            f.write("1. Review the transfer recommendations in TRANSFER_PLAN.csv\n")
            f.write(
                "2. Check the inventory status overview in INVENTORY_OVERVIEW.png\n"
            )
            f.write("3. Implement approved transfers\n")
            f.write("4. Monitor results and re-run analysis weekly\n")

    def _create_key_visualizations(
        self, analysis_df, transfer_plans, impact_data, stores_df
    ):
        """Create only the most essential visualizations."""

        # 1. Inventory Status Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Inventory Optimization Overview", fontsize=16, fontweight="bold")

        # Inventory status distribution
        status_counts = analysis_df["inventory_status"].value_counts()
        colors = [self.colors["excess"], self.colors["balanced"], self.colors["needed"]]
        ax1.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct="%1.1f%%",
            colors=colors[: len(status_counts)],
            startangle=90,
        )
        ax1.set_title("Current Inventory Status")

        # Inventory status by store
        store_status = (
            analysis_df.groupby(["store_id", "inventory_status"])
            .size()
            .unstack(fill_value=0)
        )
        store_status.plot(
            kind="bar",
            stacked=True,
            ax=ax2,
            color=[
                self.colors["balanced"],
                self.colors["excess"],
                self.colors["needed"],
            ],
        )
        ax2.set_title("Inventory Status by Store")
        ax2.set_xlabel("Store ID")
        ax2.set_ylabel("Number of Items")
        ax2.legend(title="Status")
        ax2.tick_params(axis="x", rotation=45)

        # Algorithm comparison (if available)
        if transfer_plans and len(transfer_plans) > 1:
            algorithms = list(transfer_plans.keys())
            transfer_counts = [
                len(plan) if not plan.empty else 0 for plan in transfer_plans.values()
            ]

            ax3.bar(algorithms, transfer_counts, color=self.colors["transfer"])
            ax3.set_title("Transfer Volume by Algorithm")
            ax3.set_ylabel("Number of Transfers")
            ax3.tick_params(axis="x", rotation=45)
        else:
            ax3.text(
                0.5,
                0.5,
                "Single Algorithm\nUsed",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("Algorithm Usage")

        # Store performance (items needing attention)
        store_issues = (
            analysis_df[analysis_df["inventory_status"] != "Balanced"]
            .groupby("store_id")
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        if not store_issues.empty:
            ax4.barh(
                range(len(store_issues)),
                store_issues.values,
                color=self.colors["excess"],
            )
            ax4.set_yticks(range(len(store_issues)))
            ax4.set_yticklabels([f"Store {sid}" for sid in store_issues.index])
            ax4.set_xlabel("Items Needing Attention")
            ax4.set_title("Top 10 Stores with Inventory Issues")
        else:
            ax4.text(
                0.5,
                0.5,
                "All Stores\nWell Balanced",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Store Performance")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "INVENTORY_OVERVIEW.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _generate_transfer_recommendations(
        self, transfer_plans, stores_df, products_df
    ):
        """Generate a clean, actionable transfer plan."""

        # Use the best algorithm's plan or the first available one
        if len(transfer_plans) == 1:
            plan = list(transfer_plans.values())[0]
        else:
            # Use rule-based as default, or first available
            plan = transfer_plans.get("Rule-Based", list(transfer_plans.values())[0])

        if plan.empty:
            return

        # Create store and product name mappings
        store_names = stores_df.set_index("store_id")["store_name"].to_dict()
        product_names = products_df.set_index("product_id")["product_name"].to_dict()

        # Create clean transfer plan
        clean_plan = plan.copy()
        if "from_store_id" in clean_plan.columns:
            clean_plan["From_Store"] = clean_plan["from_store_id"].map(store_names)
        if "to_store_id" in clean_plan.columns:
            clean_plan["To_Store"] = clean_plan["to_store_id"].map(store_names)
        if "product_id" in clean_plan.columns:
            clean_plan["Product"] = clean_plan["product_id"].map(product_names)

        # Select and rename key columns
        output_columns = []
        column_mapping = {
            "From_Store": "From Store",
            "To_Store": "To Store",
            "Product": "Product Name",
            "units": "Units to Transfer",
            "transport_cost": "Transport Cost ($)",
            "distance": "Distance (km)",
        }

        for col in column_mapping.keys():
            if col in clean_plan.columns:
                output_columns.append(col)

        if output_columns:
            final_plan = clean_plan[output_columns].copy()
            final_plan = final_plan.rename(columns=column_mapping)

            # Sort by cost (descending) to prioritize high-value transfers
            if "Transport Cost ($)" in final_plan.columns:
                final_plan = final_plan.sort_values(
                    "Transport Cost ($)", ascending=False
                )

            # Add priority ranking
            final_plan.insert(0, "Priority", range(1, len(final_plan) + 1))

            # Save clean transfer plan
            output_path = os.path.join(self.output_dir, "TRANSFER_PLAN.csv")
            final_plan.to_csv(output_path, index=False)

            # Create transfer summary
            self._create_transfer_summary(final_plan)

    def _create_transfer_summary(self, transfer_plan):
        """Create a simple transfer summary."""
        summary_path = os.path.join(self.output_dir, "TRANSFER_SUMMARY.txt")

        with open(summary_path, "w") as f:
            f.write("TRANSFER PLAN SUMMARY\n")
            f.write("=" * 30 + "\n\n")

            f.write(f"Total Transfers Recommended: {len(transfer_plan)}\n")

            if "Transport Cost ($)" in transfer_plan.columns:
                total_cost = transfer_plan["Transport Cost ($)"].sum()
                avg_cost = transfer_plan["Transport Cost ($)"].mean()
                f.write(f"Total Transport Cost: ${total_cost:,.2f}\n")
                f.write(f"Average Cost per Transfer: ${avg_cost:,.2f}\n")

            if "Units to Transfer" in transfer_plan.columns:
                total_units = transfer_plan["Units to Transfer"].sum()
                f.write(f"Total Units to Transfer: {total_units:,}\n")

            if "Distance (km)" in transfer_plan.columns:
                avg_distance = transfer_plan["Distance (km)"].mean()
                f.write(f"Average Transfer Distance: {avg_distance:.1f} km\n")

            f.write("\nTop 5 Priority Transfers:\n")
            f.write("-" * 25 + "\n")

            for i, row in transfer_plan.head(5).iterrows():
                f.write(
                    f"{row['Priority']}. {row.get('From Store', 'N/A')} → {row.get('To Store', 'N/A')}\n"
                )
                if "Product Name" in row:
                    f.write(f"   Product: {row['Product Name']}\n")
                if "Units to Transfer" in row:
                    f.write(f"   Units: {row['Units to Transfer']}\n")
                if "Transport Cost ($)" in row:
                    f.write(f"   Cost: ${row['Transport Cost ($)']:,.2f}\n")
                f.write("\n")

    def clean_results_directory(self):
        """Remove unnecessary files to keep only essential results."""
        files_to_keep = {
            "EXECUTIVE_SUMMARY.txt",
            "INVENTORY_OVERVIEW.png",
            "TRANSFER_PLAN.csv",
            "TRANSFER_SUMMARY.txt",
        }

        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename not in files_to_keep and not filename.startswith("."):
                    file_path = os.path.join(self.output_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"Removed: {filename}")
                    except Exception as e:
                        print(f"Could not remove {filename}: {e}")

    def print_quick_summary(self):
        """Print a quick summary to console."""
        summary_path = os.path.join(self.output_dir, "EXECUTIVE_SUMMARY.txt")
        if os.path.exists(summary_path):
            print("\n" + "=" * 60)
            print("QUICK SUMMARY")
            print("=" * 60)
            with open(summary_path, "r") as f:
                content = f.read()
                # Extract key lines for quick view
                lines = content.split("\n")
                for line in lines:
                    if any(
                        keyword in line
                        for keyword in [
                            "Items with Excess",
                            "Items Needing More",
                            "Imbalance Severity",
                            "Best Optimization",
                            "Recommended Transfers",
                            "Estimated Transport Cost",
                        ]
                    ):
                        print(line)
            print("=" * 60)
            print("Check 'results' folder for detailed files:")
            print("   • EXECUTIVE_SUMMARY.txt - Complete overview")
            print("   • INVENTORY_OVERVIEW.png - Visual dashboard")
            print("   • TRANSFER_PLAN.csv - Actionable transfer list")
            print("   • TRANSFER_SUMMARY.txt - Transfer details")
            print("=" * 60)
