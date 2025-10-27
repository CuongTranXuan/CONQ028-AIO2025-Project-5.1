# Inventory Transfer Optimization - Algorithm Logic Documentation

This document explains the logic and mathematical foundations of both optimization algorithms used in the inventory transfer system.

## Table of Contents

1. [Rule-Based Algorithm](#rule-based-algorithm)
2. [Genetic Algorithm](#genetic-algorithm)
3. [Algorithm Comparison](#algorithm-comparison)
4. [Hand Calculation Examples](#hand-calculation-examples)

---

## Rule-Based Algorithm

### Overview

The Rule-Based algorithm is a **greedy, deterministic approach** that follows simple business logic to generate transfer plans. It prioritizes practicality and interpretability over mathematical optimality.

### Core Logic Flow

```
1. Sort Data by Priority
   ├── Excess Inventory: Sort by excess_units (DESC)
   └── Needed Inventory: Sort by needed_units (DESC)

2. For Each Product in Need:
   ├── Find all stores with excess of this product
   ├── Sort excess stores by distance (ASC - closest first)
   └── Transfer from closest available stores

3. Calculate Transfer Amount:
   transfer_units = min(needed_units, available_excess_units)

4. Update Tracking:
   ├── Update transferred_from[store, product]
   └── Update transferred_to[store, product]

5. Repeat until all needs processed
```

### Detailed Algorithm Steps

#### Step 1: Data Preparation

```python
# Sort excess inventory (largest excess first)
excess_sorted = excess_inventory.sort_values("excess_units", ascending=False)

# Sort needed inventory (highest need first)
needed_sorted = needed_inventory.sort_values("needed_units", ascending=False)
```

#### Step 2: Distance-Based Matching

```python
for need_row in needed_sorted:
    # Find excess inventory for this product
    excess_for_product = excess_sorted[excess_sorted["product_id"] == need_product_id]

    # Add distance and sort by closest first
    excess_for_product["distance"] = excess_for_product["store_id"].apply(
        lambda x: distance_matrix.loc[x, need_store_id]
    )
    excess_for_product = excess_for_product.sort_values("distance")
```

#### Step 3: Transfer Calculation

```python
# Transfer from closest stores with excess
for excess_row in excess_for_product:
    transfer_units = min(needed_units, available_excess_units)

    if transfer_units > 0:
        transport_cost = cost_matrix.loc[from_store, to_store] * transfer_units

        # Record transfer
        transfers.append({
            "from_store_id": from_store,
            "to_store_id": to_store,
            "product_id": product_id,
            "units": transfer_units,
            "transport_cost": transport_cost
        })
```

### Rule-Based Algorithm Characteristics

**Advantages:**

- **Simple & Fast**: O(n×m) complexity
- **Interpretable**: Clear reasoning for each decision
- **Deterministic**: Same input → Same output
- **Practical**: Follows intuitive business logic

**Limitations:**

- **Local Optimum**: May miss globally optimal solution
- **Greedy**: Early decisions can prevent better later solutions
- **No Backtracking**: Cannot undo suboptimal transfers

---

## Genetic Algorithm

### Overview

The Genetic Algorithm mimics **natural evolution** to find optimal transfer plans. It maintains a population of candidate solutions that evolve over generations through selection, crossover, and mutation.

### Core Concepts

#### Individual (Chromosome)

```python
class Individual:
    transfer_plan = [
        {"from_store": 1, "to_store": 3, "product_id": 5, "units": 10},
        {"from_store": 2, "to_store": 4, "product_id": 7, "units": 15},
        # ... more transfers
    ]
    fitness = 45000  # Total transport cost (lower is better)
```

#### Population

```
Generation 0: [Individual₁, Individual₂, ..., Individual₅₀]
              [Fitness₁,   Fitness₂,   ..., Fitness₅₀]
```

### Detailed Algorithm Flow

#### Step 1: Initialize Population

```python
def create_initial_population(population_size):
    population = []
    for i in range(population_size):
        individual = create_random_solution()
        population.append(individual)
    return population

def create_random_solution():
    transfer_plan = []
    for product_id in valid_products:
        # Randomly match excess stores to needed stores
        # Random transfer amounts within constraints
        transfer_plan.append(random_transfer)
    return transfer_plan
```

#### Step 2: Fitness Evaluation

```python
def calculate_fitness(transfer_plan):
    total_cost = 0
    for transfer in transfer_plan:
        cost_per_unit = transport_cost_matrix[from_store, to_store]
        total_cost += cost_per_unit * units
    return total_cost  # Lower = Better
```

#### Step 3: Selection (Tournament Selection)

```python
def tournament_selection(population, tournament_size=3):
    parents = []
    for _ in range(len(population)):
        # Pick random individuals for tournament
        tournament = random.sample(population, tournament_size)
        # Winner = best fitness (lowest cost)
        winner = min(tournament, key=lambda x: x.fitness)
        parents.append(winner)
    return parents
```

#### Step 4: Crossover (Single-Point)

```python
def crossover(parent1, parent2):
    if random.random() < crossover_probability:
        # Choose crossover point
        point = random.randint(1, min(len(parent1), len(parent2)) - 1)

        # Create children by swapping segments
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        # Repair constraint violations
        child1 = repair_solution(child1)
        child2 = repair_solution(child2)
    else:
        child1, child2 = parent1.copy(), parent2.copy()

    return child1, child2
```

#### Step 5: Mutation

```python
def mutate(individual):
    if random.random() < mutation_probability:
        mutation_type = random.choice([
            "change_quantity",    # Modify transfer amount
            "remove_transfer",    # Delete a transfer
            "add_transfer",       # Add new transfer
            "change_route"        # Change source/destination
        ])
        apply_mutation(individual, mutation_type)
        individual = repair_solution(individual)
```

#### Step 6: Evolution Loop

```python
for generation in range(num_generations):
    # Selection
    parents = tournament_selection(population)

    # Crossover & Mutation
    offspring = []
    for i in range(0, len(parents), 2):
        child1, child2 = crossover(parents[i], parents[i+1])
        mutate(child1)
        mutate(child2)
        offspring.extend([child1, child2])

    # Evaluation
    evaluate_population(offspring)

    # Elitism: Keep best from previous generation
    best_previous = min(population, key=lambda x: x.fitness)
    worst_current = max(offspring, key=lambda x: x.fitness)
    if best_previous.fitness < worst_current.fitness:
        offspring[offspring.index(worst_current)] = best_previous

    # Replace population
    population = offspring
```

### Genetic Algorithm Parameters

| Parameter             | Typical Value | Description                          |
| --------------------- | ------------- | ------------------------------------ |
| Population Size       | 50-100        | Number of individuals per generation |
| Generations           | 100-500       | Number of evolution cycles           |
| Crossover Probability | 0.8-0.9       | Chance of crossover between parents  |
| Mutation Probability  | 0.1-0.2       | Chance of mutation in offspring      |
| Tournament Size       | 3-5           | Individuals competing in selection   |

---

## Algorithm Comparison

| Aspect               | Rule-Based                       | Genetic Algorithm                                     |
| -------------------- | -------------------------------- | ----------------------------------------------------- |
| **Approach**         | Greedy, Deterministic            | Evolutionary, Stochastic                              |
| **Solution Quality** | Good (Local Optimum)             | Better (Near-Global Optimum)                          |
| **Execution Time**   | Fast (seconds)                   | Slower (minutes)                                      |
| **Consistency**      | Always same result               | Varies between runs                                   |
| **Interpretability** | High (clear rules)               | Low (complex evolution)                               |
| **Scalability**      | O(n×m)                           | O(g×p×f) where g=generations, p=population, f=fitness |
| **Best Use Case**    | Quick decisions, simple problems | Complex optimization, quality critical                |

---

## Hand Calculation Examples

### Example Scenario

**Stores:** A, B, C, D (IDs: 1, 2, 3, 4)
**Products:** X, Y (IDs: 5, 6)

**Inventory Data:**

```
Excess Inventory:
Store A: Product X = 20 excess units
Store B: Product Y = 15 excess units

Needed Inventory:
Store C: Product X = 12 needed units
Store D: Product X = 8 needed units
Store D: Product Y = 10 needed units
```

**Distance Matrix (km):**

```
     A   B   C   D
A    0  10  25  15
B   10   0  20  30
C   25  20   0  12
D   15  30  12   0
```

**Transport Cost Matrix (VND per unit):**

```
     A      B      C      D
A    0   1000   2500   1500
B 1000      0   2000   3000
C 2500   2000      0   1200
D 1500   3000   1200      0
```

### Rule-Based Calculation

#### Step 1: Sort by Priority

```
Needed (sorted by needed_units DESC):
1. Store C: Product X = 12 units needed
2. Store D: Product Y = 10 units needed
3. Store D: Product X = 8 units needed

Excess (sorted by excess_units DESC):
1. Store A: Product X = 20 units excess
2. Store B: Product Y = 15 units excess
```

#### Step 2: Process Each Need

**Need 1: Store C needs 12 units of Product X**

- Available excess: Store A has 20 units of Product X
- Distance A→C: 25 km
- Transfer: 12 units from A to C
- Cost: 2500 VND/unit × 12 units = 30,000 VND

**Need 2: Store D needs 10 units of Product Y**

- Available excess: Store B has 15 units of Product Y
- Distance B→D: 30 km
- Transfer: 10 units from B to D
- Cost: 3000 VND/unit × 10 units = 30,000 VND

**Need 3: Store D needs 8 units of Product X**

- Available excess: Store A has 8 remaining units (20-12=8)
- Distance A→D: 15 km
- Transfer: 8 units from A to D
- Cost: 1500 VND/unit × 8 units = 12,000 VND

#### Rule-Based Result:

```
Transfers:
1. A → C: Product X, 12 units, 30,000 VND
2. B → D: Product Y, 10 units, 30,000 VND
3. A → D: Product X, 8 units, 12,000 VND

Total Cost: 72,000 VND
Total Transfers: 3
Total Units: 30
```

### Genetic Algorithm Calculation

#### Step 1: Individual Representation

```
Individual₁ = [
    {"from": A, "to": C, "product": X, "units": 12},
    {"from": B, "to": D, "product": Y, "units": 10},
    {"from": A, "to": D, "product": X, "units": 8}
]

Individual₂ = [
    {"from": A, "to": D, "product": X, "units": 20},
    {"from": B, "to": D, "product": Y, "units": 10}
]
```

#### Step 2: Fitness Calculation

**Individual₁ Fitness:**

```
Transfer 1: A→C, Product X, 12 units
Cost = 2500 × 12 = 30,000 VND

Transfer 2: B→D, Product Y, 10 units
Cost = 3000 × 10 = 30,000 VND

Transfer 3: A→D, Product X, 8 units
Cost = 1500 × 8 = 12,000 VND

Total Fitness₁ = 72,000 VND
```

**Individual₂ Fitness:**

```
Transfer 1: A→D, Product X, 20 units
Cost = 1500 × 20 = 30,000 VND

Transfer 2: B→D, Product Y, 10 units
Cost = 3000 × 10 = 30,000 VND

Total Fitness₂ = 60,000 VND
```

_Note: Individual₂ is better (lower cost) but doesn't satisfy Store C's need_

#### Step 3: Selection Example (Tournament Size=2)

```
Tournament 1: [Individual₁, Individual₂]
Winner: Individual₂ (lower fitness: 60,000 < 72,000)

Tournament 2: [Individual₃, Individual₄]
Winner: Individual with lowest fitness
```

#### Step 4: Crossover Example

```
Parent₁ = [Transfer_A, Transfer_B, Transfer_C]
Parent₂ = [Transfer_X, Transfer_Y]

Crossover Point = 1

Child₁ = [Transfer_A] + [Transfer_Y] = [Transfer_A, Transfer_Y]
Child₂ = [Transfer_X] + [Transfer_B, Transfer_C] = [Transfer_X, Transfer_B, Transfer_C]

After Repair (ensure constraints):
Child₁ = Valid transfers only
Child₂ = Valid transfers only
```

#### Step 5: Mutation Example

```
Original: [A→C: X,12] [B→D: Y,10] [A→D: X,8]

Mutation Type: Change Quantity
Target: Transfer 1 (A→C: X,12)
New Quantity: Random between 1 and min(20_available, 12_needed) = Random(1,12) = 10

Mutated: [A→C: X,10] [B→D: Y,10] [A→D: X,8]
```

### Performance Comparison for Example

| Algorithm        | Total Cost | Transfers | Satisfied Needs    | Execution       |
| ---------------- | ---------- | --------- | ------------------ | --------------- |
| Rule-Based       | 72,000 VND | 3         | 100% (30/30 units) | Deterministic   |
| GA (Individual₂) | 60,000 VND | 2         | 67% (20/30 units)  | Need repair     |
| GA (Optimized)   | 65,000 VND | 3         | 100% (30/30 units) | After evolution |

_The GA might find alternative solutions like splitting transfers differently or finding more efficient routing that the rule-based approach missed due to its greedy nature._
