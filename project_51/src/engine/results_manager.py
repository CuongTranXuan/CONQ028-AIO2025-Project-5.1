"""
Results Manager for Inventory Optimization System.

This module handles the creation of simplified result files.
"""

import os

import pandas as pd


class ResultsManager:
    def __init__(self, results_dir):
        """
        Initialize Results Manager.

        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def create_final_results(self, results_dict, stores_df, products_df):
        """
        Create the final simplified results: summary and best transfer plan.

        Args:
            results_dict: Dictionary with algorithm results (algorithm_name: (transfer_plan, impact_df))
            stores_df: Store information DataFrame
            products_df: Product information DataFrame
        """
        if not results_dict:
            print("No optimization results to process.")
            return

        # Find best algorithm and create results
        best_algorithm = self._find_best_algorithm(results_dict)

        # Create result summary
        self._create_result_summary(results_dict, best_algorithm)

        # Create best transfer plan CSV
        self._create_best_transfer_plan_csv(
            results_dict, best_algorithm, stores_df, products_df
        )

        print(f"\nResults created:")
        print(f"  • result_summary.txt - Overall performance summary")
        print(f"  • best_transfer_plan.csv - Best algorithm transfer plan")

    def _find_best_algorithm(self, results_dict):
        """
        Find the best performing algorithm based on key metrics.

        Args:
            results_dict: Dictionary with algorithm results

        Returns:
            Name of the best algorithm
        """
        if len(results_dict) == 1:
            return list(results_dict.keys())[0]

        algorithm_scores = {}

        for algorithm, (transfer_plan, impact_df) in results_dict.items():
            score = 0

            # Skip if no transfer plan
            if transfer_plan is None or transfer_plan.empty:
                algorithm_scores[algorithm] = 0
                continue

            # Base score for having transfers
            score += 10

            # Cost efficiency (lower cost per unit is better)
            if (
                "transport_cost" in transfer_plan.columns
                and "units" in transfer_plan.columns
            ):
                total_cost = transfer_plan["transport_cost"].sum()
                total_units = transfer_plan["units"].sum()
                if total_units > 0:
                    cost_per_unit = total_cost / total_units
                    # Normalize cost score (lower is better)
                    score += max(0, 100 - (cost_per_unit / 10000))

            # Inventory improvement score
            if (
                impact_df is not None
                and hasattr(impact_df, "columns")
                and "Improvement" in impact_df.columns
            ):
                try:
                    improvement_value = impact_df.loc[
                        "Increase in Balanced Items", "Improvement"
                    ]
                    score += improvement_value * 2  # Weight improvement highly
                except (KeyError, IndexError):
                    pass

            # Efficiency score (more units transferred is better, fewer transfers is better)
            if "units" in transfer_plan.columns:
                total_units = transfer_plan["units"].sum()
                num_transfers = len(transfer_plan)
                if num_transfers > 0:
                    efficiency = total_units / num_transfers
                    score += min(efficiency / 10, 20)  # Cap at 20 points

            algorithm_scores[algorithm] = score

        # Return algorithm with highest score
        return max(algorithm_scores.keys(), key=lambda k: algorithm_scores[k])

    def _create_result_summary(self, results_dict, best_algorithm):
        """
        Create result_summary.txt with algorithm comparison and best algorithm info.

        Args:
            results_dict: Dictionary with algorithm results
            best_algorithm: Name of the best algorithm
        """
        summary_path = os.path.join(self.results_dir, "result_summary.txt")

        with open(summary_path, "w") as f:
            f.write("INVENTORY OPTIMIZATION RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Algorithm comparison
            f.write("ALGORITHM COMPARISON:\n")
            f.write("-" * 30 + "\n")

            for algorithm, (transfer_plan, impact_df) in results_dict.items():
                marker = "[BEST]" if algorithm == best_algorithm else "      "
                f.write(f"\n{marker} {algorithm} Algorithm:\n")

                if transfer_plan is not None and not transfer_plan.empty:
                    f.write(f"   Status: [OK] Completed Successfully\n")
                    f.write(f"   Transfers Generated: {len(transfer_plan)}\n")

                    if "transport_cost" in transfer_plan.columns:
                        total_cost = transfer_plan["transport_cost"].sum()
                        f.write(f"   Total Transport Cost: ${total_cost:,.2f}\n")

                    if "units" in transfer_plan.columns:
                        total_units = transfer_plan["units"].sum()
                        f.write(f"   Total Units to Transfer: {total_units:,}\n")

                        # Calculate efficiency
                        if len(transfer_plan) > 0:
                            efficiency = total_units / len(transfer_plan)
                            f.write(
                                f"   Average Units per Transfer: {efficiency:.1f}\n"
                            )

                    # Impact information
                    if (
                        impact_df is not None
                        and hasattr(impact_df, "columns")
                        and "Improvement" in impact_df.columns
                    ):
                        try:
                            improvement_value = impact_df.loc[
                                "Increase in Balanced Items", "Improvement"
                            ]
                            f.write(
                                f"   Inventory Improvement: {improvement_value} items\n"
                            )
                        except (KeyError, IndexError):
                            pass
                else:
                    f.write(f"   Status: [FAILED] No transfers generated\n")

            # Best algorithm summary
            f.write(f"\n\nRECOMMENDED SOLUTION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Best Algorithm: {best_algorithm}\n")

            best_plan, best_impact = results_dict[best_algorithm]
            if best_plan is not None and not best_plan.empty:
                f.write(f"Recommended Transfers: {len(best_plan)}\n")

                if "transport_cost" in best_plan.columns:
                    total_cost = best_plan["transport_cost"].sum()
                    f.write(f"Total Cost: ${total_cost:,.2f}\n")

                if "units" in best_plan.columns:
                    total_units = best_plan["units"].sum()
                    f.write(f"Total Units: {total_units:,}\n")

    def _create_best_transfer_plan_csv(
        self, results_dict, best_algorithm, stores_df, products_df
    ):
        """
        Create best_transfer_plan.csv with the best algorithm's transfer plan.

        Args:
            results_dict: Dictionary with algorithm results
            best_algorithm: Name of the best algorithm
            stores_df: Store information DataFrame
            products_df: Product information DataFrame
        """
        best_plan, _ = results_dict[best_algorithm]

        if best_plan is None or best_plan.empty:
            print(
                f"Warning: Best algorithm '{best_algorithm}' has no transfer plan to save."
            )
            return

        # Create store and product name mappings
        store_names = stores_df.set_index("store_id")["store_name"].to_dict()
        product_names = products_df.set_index("product_id")["product_name"].to_dict()

        # Enhance transfer plan with readable names
        enhanced_plan = best_plan.copy()

        # Add readable names
        if "from_store_id" in enhanced_plan.columns:
            enhanced_plan["from_store_name"] = enhanced_plan["from_store_id"].map(
                store_names
            )
        if "to_store_id" in enhanced_plan.columns:
            enhanced_plan["to_store_name"] = enhanced_plan["to_store_id"].map(
                store_names
            )
        if "product_id" in enhanced_plan.columns:
            enhanced_plan["product_name"] = enhanced_plan["product_id"].map(
                product_names
            )

        # Reorder columns for better readability
        column_order = []

        # Priority columns first
        if "from_store_name" in enhanced_plan.columns:
            column_order.append("from_store_name")
        if "to_store_name" in enhanced_plan.columns:
            column_order.append("to_store_name")
        if "product_name" in enhanced_plan.columns:
            column_order.append("product_name")
        if "units" in enhanced_plan.columns:
            column_order.append("units")
        if "transport_cost" in enhanced_plan.columns:
            column_order.append("transport_cost")
        if "distance_km" in enhanced_plan.columns:
            column_order.append("distance_km")

        # Add remaining columns
        for col in enhanced_plan.columns:
            if col not in column_order:
                column_order.append(col)

        # Reorder and save
        final_plan = enhanced_plan[column_order]

        # Sort by transport cost (highest priority first)
        if "transport_cost" in final_plan.columns:
            final_plan = final_plan.sort_values("transport_cost", ascending=False)

        # Save to CSV
        csv_path = os.path.join(self.results_dir, "best_transfer_plan.csv")
        final_plan.to_csv(csv_path, index=False)

        print(
            f"Best transfer plan saved: {len(final_plan)} transfers from {best_algorithm} algorithm"
        )
