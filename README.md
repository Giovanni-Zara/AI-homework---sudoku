# Sudoku Solver - AI Project

A Sudoku puzzle solver implementing and comparing two AI approaches: **A\* Search** and **Constraint Satisfaction Problem (CSP)**.

## Overview

This project solves Sudoku puzzles using two different algorithms:

1. **A\* Search**: Graph-based search with heuristics (empty cells count and advanced constraint-based)
2. **CSP Solver**: Reduces Sudoku to a constraint satisfaction problem using `AllDifferent` constraints

The project includes benchmarking tools to compare performance across puzzles of varying difficulty.

## Project Structure

| File | Description |
|------|-------------|
| `sudoku.py` | Core Sudoku representation and state management |
| `a_star.py` | A\* search algorithm implementation |
| `sudoku_csp.py` | CSP-based solver using python-constraint |
| `benchmark.py` | Performance comparison and metrics collection |
| `benchmark_results.csv` | Pre-computed benchmark results |

## Dependencies

```
numpy
python-constraint
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Solve with A\* Search

```bash
python a_star.py
```

### Solve with CSP

```bash
python sudoku_csp.py
```

### Run Benchmarks

```bash
python benchmark.py
```

This runs both algorithms on 8 puzzles (easy to expert) and exports results to `benchmark_results.csv`.

## CSP Formulation

- **Variables**: 81 cells (`cell_r_c` for row r, column c)
- **Domains**: `{1-9}` for empty cells, `{fixed_value}` for pre-filled
- **Constraints**: 27 `AllDifferent` constraints (9 rows + 9 columns + 9 boxes)

## Benchmark Results Summary

| Difficulty | A\* (empty cells) | A\* (advanced) | CSP |
|------------|-------------------|----------------|-----|
| Easy | ~9-18 ms | ~18 ms | ~2-4 ms |
| Medium | ~19-47 ms | ~25-47 ms | ~4-6 ms |
| Hard | ~75-2600 ms | ~31-3600 ms | ~3-301 ms |
| Expert | ~245-1300 ms | ~195-3200 ms | ~44-2400 ms |

## Author

AI Course Homework - Artificial Intelligence, Master's Degree (1st Year)
