import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# Import configuration
from src.config import (
    DATA_DIR,
    EXCESS_PERCENT,
    GA_CROSSOVER_PROB,
    GA_GENERATIONS,
    GA_MUTATION_PROB,
    GA_POPULATION_SIZE,
    LOGS_DIR,
    MAX_INVENTORY_DAYS,
    MIN_INVENTORY_DAYS,
    NUM_PRODUCTS,
    RANDOM_SEED,
    REQUIRED_DATA_FILES,
    RESULTS_DIR,
    SALES_DAYS,
    SHORTAGE_PERCENT,
    VISUALIZATIONS_DIR,
    create_directories,
    get_ga_config,
)

# Import components
from src.data_generator.data_generator_main import generate_all_data
from src.engine.analyzer import InventoryAnalyzer
from src.engine.genetic_algorithm import GeneticAlgorithmOptimizer
from src.engine.results_manager import ResultsManager
from src.engine.rule_based import RuleBasedOptimizer


def setup_directories():
    """Create necessary directories using config."""
    return create_directories()


def run_data_generation(args):
    """Run data generation."""
    print("\n=== DATA GENERATION ===")
    generate_all_data(
        num_products=args.products,
        days=args.days,
        output_dir=args.data_dir,
        random_seed=args.seed,
        min_days=args.min_days,
        max_days=args.max_days,
        excess_percent=args.excess_percent,
        shortage_percent=args.shortage_percent,
    )


def run_analysis(args):
    """Run inventory analysis."""
    print("\n=== INVENTORY ANALYSIS ===")

    # Create analyzer
    analyzer = InventoryAnalyzer()

    # Load data
    analyzer.load_data(
        sales_path=os.path.join(args.data_dir, "sales_data.csv"),
        inventory_path=os.path.join(args.data_dir, "inventory_data.csv"),
        stores_path=os.path.join(args.data_dir, "stores.csv"),
        products_path=os.path.join(args.data_dir, "products.csv"),
    )

    # Analyze data
    analysis_df = analyzer.analyze_sales_data()

    # Identify imbalances
    excess_df, needed_df = analyzer.identify_inventory_imbalances(
        min_days=args.min_days, max_days=args.max_days
    )

    # Save analysis results
    analysis_df.to_csv(
        os.path.join(args.results_dir, "inventory_analysis.csv"), index=False
    )
    excess_df.to_csv(
        os.path.join(args.results_dir, "excess_inventory.csv"), index=False
    )
    needed_df.to_csv(
        os.path.join(args.results_dir, "needed_inventory.csv"), index=False
    )

    # Calculate total excess and needed units
    excess_units = excess_df["excess_units"].sum()
    needed_units = needed_df["needed_units"].sum()

    print(f"\nTotal excess units: {excess_units}")
    print(f"Total needed units: {needed_units}")
    print(f"Excess to needed ratio: {excess_units / needed_units:.2f}")

    return analyzer, analysis_df, excess_df, needed_df


def run_rule_based_optimization(analyzer, excess_df, needed_df, args):
    """Run rule-based optimization."""
    print("\n=== RULE-BASED OPTIMIZATION ===")

    # Create optimizer
    optimizer = RuleBasedOptimizer()

    # Load matrices
    optimizer.load_matrices(
        distance_path=os.path.join(args.data_dir, "distance_matrix.csv"),
        cost_path=os.path.join(args.data_dir, "transport_cost_matrix.csv"),
    )

    # Measure execution time
    start_time = time.time()

    # Generate transfer plan
    transfer_plan = optimizer.optimize(excess_df, needed_df)

    execution_time = time.time() - start_time
    print(f"Rule-based optimization completed in {execution_time:.2f} seconds")

    # Add store and product names
    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))
    optimizer.add_store_product_names(stores_df, products_df)

    # Save transfer plan
    if not transfer_plan.empty:
        transfer_plan.to_csv(
            os.path.join(args.results_dir, "rule_based_transfers.csv"), index=False
        )

        # Evaluate impact
        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)

        # Save impact analysis
        pd.DataFrame(impact_df).to_csv(
            os.path.join(args.results_dir, "rule_based_impact.csv")
        )

        return transfer_plan, impact_df

    return transfer_plan, None


def run_ga_optimization(analyzer, excess_df, needed_df, args):
    """Run genetic algorithm optimization."""
    print("\n=== GENETIC ALGORITHM OPTIMIZATION ===")

    # Create optimizer
    optimizer = GeneticAlgorithmOptimizer(random_seed=args.seed)

    # Load matrices
    optimizer.load_matrices(
        distance_path=os.path.join(args.data_dir, "distance_matrix.csv"),
        cost_path=os.path.join(args.data_dir, "transport_cost_matrix.csv"),
    )

    # Measure execution time
    start_time = time.time()

    # Generate transfer plan
    transfer_plan = optimizer.optimize(
        excess_df,
        needed_df,
        population_size=args.ga_population,
        num_generations=args.ga_generations,
        crossover_prob=args.ga_crossover,
        mutation_prob=args.ga_mutation,
    )

    execution_time = time.time() - start_time
    print(f"Genetic algorithm optimization completed in {execution_time:.2f} seconds")

    # Add store and product names
    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))
    optimizer.add_store_product_names(stores_df, products_df)

    # Save transfer plan
    if not transfer_plan.empty:
        transfer_plan.to_csv(
            os.path.join(args.results_dir, "ga_transfers.csv"), index=False
        )

        # Evaluate impact
        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)

        # Save impact analysis
        pd.DataFrame(impact_df).to_csv(os.path.join(args.results_dir, "ga_impact.csv"))

        return transfer_plan, impact_df

    return transfer_plan, None


def create_results(analysis_df, results_dict, analyzer, args):
    """Create simplified results: summary and best transfer plan."""
    print("\n=== GENERATING RESULTS ===")

    # Load store and product data
    stores_df = pd.read_csv(os.path.join(args.data_dir, "stores.csv"))
    products_df = pd.read_csv(os.path.join(args.data_dir, "products.csv"))

    # Create results manager and generate final results
    results_manager = ResultsManager(args.results_dir)
    results_manager.create_final_results(results_dict, stores_df, products_df)


def main():
    parser = argparse.ArgumentParser(
        description="Inventory Transfer Optimization System"
    )

    # General options
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument(
        "--results-dir", type=str, default=RESULTS_DIR, help="Results directory"
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=VISUALIZATIONS_DIR,
        help="Visualizations directory",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")

    # Data generation options
    parser.add_argument("--generate-data", action="store_true", help="Generate data")
    parser.add_argument(
        "--products", type=int, default=NUM_PRODUCTS, help="Number of products"
    )
    parser.add_argument(
        "--days", type=int, default=SALES_DAYS, help="Number of days of sales data"
    )
    parser.add_argument(
        "--excess-percent",
        type=int,
        default=EXCESS_PERCENT,
        help="Percentage of items with excess inventory",
    )
    parser.add_argument(
        "--shortage-percent",
        type=int,
        default=SHORTAGE_PERCENT,
        help="Percentage of items with shortage",
    )

    # Analysis options
    parser.add_argument(
        "--min-days",
        type=int,
        default=MIN_INVENTORY_DAYS,
        help="Minimum days of inventory",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=MAX_INVENTORY_DAYS,
        help="Maximum days of inventory",
    )

    # Optimization options
    parser.add_argument(
        "--rule-based", action="store_true", help="Run rule-based optimization"
    )
    parser.add_argument(
        "--ga", action="store_true", help="Run genetic algorithm optimization"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all optimization methods"
    )

    # Genetic algorithm options
    parser.add_argument(
        "--ga-population",
        type=int,
        default=GA_POPULATION_SIZE,
        help="GA population size",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=GA_GENERATIONS,
        help="GA number of generations",
    )
    parser.add_argument(
        "--ga-crossover",
        type=float,
        default=GA_CROSSOVER_PROB,
        help="GA crossover probability",
    )
    parser.add_argument(
        "--ga-mutation",
        type=float,
        default=GA_MUTATION_PROB,
        help="GA mutation probability",
    )

    # Display options
    parser.add_argument(
        "--summary-only", action="store_true", help="Show only summary results"
    )

    args = parser.parse_args()

    # Create directories
    directories = setup_directories()
    args.data_dir = str(directories["data"])
    args.vis_dir = str(directories["visualizations"])
    args.results_dir = str(directories["results"])

    # Generate data if needed
    if args.generate_data:
        run_data_generation(args)

    # Check if data exists
    for file in REQUIRED_DATA_FILES:
        file_path = Path(args.data_dir) / file
        if not file_path.exists():
            print(
                f"Required file {file} not found. Please run with --generate-data first."
            )
            return

    # Run analysis
    analyzer, analysis_df, excess_df, needed_df = run_analysis(args)

    # Run optimizations
    results_dict = {}

    if args.rule_based or args.all:
        transfer_plan, impact_df = run_rule_based_optimization(
            analyzer, excess_df, needed_df, args
        )
        results_dict["Rule-Based"] = (transfer_plan, impact_df)

    if args.ga or args.all:
        transfer_plan, impact_df = run_ga_optimization(
            analyzer, excess_df, needed_df, args
        )
        results_dict["Genetic Algorithm"] = (transfer_plan, impact_df)

    # Create comprehensive results and reports
    if results_dict:
        create_results(analysis_df, results_dict, analyzer, args)

    print("\n=== INVENTORY TRANSFER OPTIMIZATION COMPLETE ===")
    print(f"Results saved to {args.results_dir} directory:")
    print(f"  • result_summary.txt - Algorithm comparison and recommendations")
    print(f"  • best_transfer_plan.csv - Optimized transfer plan")


if __name__ == "__main__":
    main()
