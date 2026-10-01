"""
run_all.py

Runs the full pipeline end-to-end:
  1. Simulate 3 distinct raw study exports -> data/raw/
  2. Harmonize + consolidate + validate into one unified dataset -> data/processed/
  3. Exploratory analyses + regulatory-style figures -> output/

Run with:  python run_all.py
"""

import analysis
import integrate
import simulate_studies


def main():
    print("--- Simulating 3 independent study datasets ---")
    simulate_studies.simulate()

    print("\n--- Harmonizing and consolidating into one unified dataset ---")
    integrate.main()

    print("\n--- Exploratory analyses and regulatory visualizations ---")
    analysis.main()


if __name__ == "__main__":
    main()
