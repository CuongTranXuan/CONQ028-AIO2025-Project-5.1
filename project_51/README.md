# Inventory Transfer Optimization System

## Overview

A system that helps retail stores manage their inventory better. It analyzes sales data and store locations to recommend which products should be moved between stores. This helps reduce excess inventory, prevent shortages, and minimize shipping costs.

### What This System Does

- **Reduces Inventory Costs**: Finds stores with too much inventory
- **Prevents Stockouts**: Identifies stores that need more products
- **Saves on Shipping**: Calculates the cheapest ways to move products
- **Makes Smart Decisions**: Uses past sales data to make recommendations

## Features

- **Inventory Analysis**: Automatically finds which stores have too much or too little inventory
- **Two Optimization Methods**:
  - **Rule-based**: Fast and simple method for quick results
  - **Genetic Algorithm**: More advanced method for better optimization
- **Cost Calculation**: Finds the cheapest routes for moving products
- **Detailed Reports**: Provides analysis and performance comparisons
- **Visual Charts**: Creates graphs and charts to show results
- **Clear Recommendations**: Gives specific instructions on what to transfer

## Project Structure

```
inventory_optimization/
├── data/                       # Input data files
│   ├── stores.csv             # Store locations and information
│   ├── products.csv           # Product catalog
│   ├── sales_data.csv         # Past sales records
│   ├── inventory_data.csv     # Current inventory levels
│   ├── distance_matrix.csv    # Distance between stores
│   └── transport_cost_matrix.csv # Transport costs
│
├── src/                        # Source code
│   ├── data_generator/        # Creates sample data
│   │   ├── store_generator.py
│   │   ├── product_generator.py
│   │   ├── sales_generator.py
│   │   ├── inventory_generator.py
│   │   └── distance_calculator.py
│   │
│   ├── engine/                # Optimization algorithms
│   │   ├── data_model.py      # Data structures
│   │   ├── analyzer.py        # Analyzes inventory
│   │   ├── rule_based.py      # Simple optimization method
│   │   └── genetic_algorithm.py # Advanced optimization method
│   │
│   ├── utils/                 # Helper functions
│   ├── config.py              # Settings
│   └── main.py                # Main program
│
├── notebooks/                 # Analysis notebooks
├── results/                   # Output files and reports
├── visualizations/            # Charts and graphs
├── logs/                      # Application logs
└── docs/                      # Documentation
```

## Quick Start

### What You Need

- Python 3.9 or newer
- Git

### 1. Installation

**Method A: Using Conda (Easier)**

```bash
# Download the code
git clone <this-repo>
cd inventory_optimization

# Set up Python environment
conda env create -f environment.yml
conda activate inventory_optimization
```

**Method B: Using pip**

```bash
# Download the code
git clone <this-repo>
cd inventory_optimization

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run the System

```bash
# Create sample data and run both optimization methods
python src/main.py --generate-data --all
```

### 3. View Results

After running, check the `results/` folder for these files:

| File                     | Description                                 |
| ------------------------ | ------------------------------------------- |
| `result_summary.txt`     | Executive summary with key metrics          |
| `best_transfer_plan.csv` | Recommended transfers to implement          |
| `inventory_analysis.csv` | Detailed inventory status by store          |
| `*_impact.csv`           | Expected impact analysis for each algorithm |

## Step-by-Step Usage Guide

### Step 1: Prepare Your Data

**Using Sample Data (for testing):**

```bash
# Create fake data for testing (30 products across multiple stores)
python src/main.py --generate-data
```

**Using Your Own Data:**
Put your CSV files in the `data/` folder with these columns:

- `stores.csv`: `store_id, name, city, latitude, longitude`
- `products.csv`: `product_id, name, category, cost, price`
- `sales_data.csv`: `store_id, product_id, date, quantity_sold`
- `inventory_data.csv`: `store_id, product_id, quantity`
- `distance_matrix.csv`: Square matrix with store distances
- `transport_cost_matrix.csv`: Square matrix with transport costs

### Step 2: Run the Optimization

**Simple Method (Fast):**

```bash
python src/main.py --rule-based
```

- Good for: Quick results, simple cases
- How it works: Uses simple rules to match stores with excess to stores that need inventory

**Advanced Method (Better Results):**

```bash
python src/main.py --ga
```

- Takes: xxx-xxx minutes
- Good for: Complex cases, best possible results
- How it works: Uses advanced algorithms to find optimal solutions

**Run Both Methods:**

```bash
python src/main.py --all
```

### Step 3: Look at Results

**Detailed Analysis:**

```bash
jupyter notebook notebooks/01_full_process_step_by_step.ipynb
```

**View Charts:**
Check the `visualizations/` folder for:

- Charts showing transfer connections (`.png`, `.graphml`)
- Summary analysis

### Step 4: Use the Recommendations

Look at `best_transfer_plan.csv` which shows:

- Which stores to move products from and to
- How many of each product to transfer
- Expected costs and benefits
- Which transfers are most important

## Advanced Settings

### Command Options

```bash
python src/main.py [OPTIONS]

Options:
  --generate-data          Create sample data for testing
  --rule-based            Use the simple optimization method
  --ga                    Use the advanced optimization method
  --all                   Run both methods and compare results
  --data-dir PATH         Where to find input data (default: data)
  --results-dir PATH      Where to save results (default: results)
  --vis-dir PATH          Where to save charts (default: visualizations)
  --seed INTEGER          Random seed for consistent results (default: 2025)
  --min-days INTEGER      Minimum days of inventory (default: 7)
  --max-days INTEGER      Maximum days of inventory (default: 21)
  --ga-population INTEGER How many solutions to try (default: 100)
  --ga-generations INTEGER How many rounds of improvement (default: 50)
```

### Settings File

Change settings in `src/config.py`:

```python
# When inventory is too low or too high
MIN_INVENTORY_DAYS = 7   # Less than this = need more inventory
MAX_INVENTORY_DAYS = 21  # More than this = too much inventory

# Advanced optimization settings
GA_POPULATION_SIZE = 100  # Number of solutions to try
GA_GENERATIONS = 50       # Number of improvement rounds
GA_CROSSOVER_PROB = 0.7   # How often to combine solutions
GA_MUTATION_PROB = 0.1    # How often to try random changes

# Sample data settings
NUM_PRODUCTS = 30         # Number of products to create
SALES_DAYS = 90          # Days of sales history
EXCESS_PERCENT = 20      # Percent of stores with too much inventory
SHORTAGE_PERCENT = 20    # Percent of stores needing inventory
```

## Understanding the Results

### Method Comparison

| Aspect             | Simple Method   | Advanced Method     |
| ------------------ | --------------- | ------------------- |
| **Speed**          | Fast (xxxs)     | Slower (xxx-xxxmin) |
| **Difficulty**     | Easy            | Complex             |
| **Result Quality** | Good            | Maybe Better        |
| **Best For**       | Quick decisions | Detailed planning   |

### What the Numbers Mean

- **Transfers Generated**: How many moves are recommended
- **Total Transport Cost**: How much shipping will cost
- **Units to Transfer**: Total number of products to move
- **Inventory Improvement**: How many inventory problems were solved

### Good Results Look Like

**Signs of Success:**

- Lower shipping costs
- More inventory problems solved
- Transfers spread across many stores
- Reasonable quantities to move

**Signs to Review:**

- Very high shipping costs
- Very few transfers suggested
- Only a few stores involved

## Detailed Analysis

### Using Jupyter Notebooks

The project includes notebooks for deeper analysis:

```bash
# Start Jupyter
jupyter notebook

# Open the analysis notebook
notebooks/01_full_process_step_by_step.ipynb
```

**What the Notebook Can Do:**

- Explore your data step by step
- Compare different methods
- Create custom charts
- Try different settings
- Export results

### Visual Charts

The system creates charts that show:

- How stores are connected
- Which transfers are recommended
- Current inventory status
- Cost relationships

Files created:

- `visualizations/store_transfer_network.graphml` (for Gephi software)
- `visualizations/network_analysis_summary.png`
