Code Pointers:
The main experiment script is provided in code/main_experiments.py.
The model implementations are provided in code/models/.
The plotting script is provided in code/plot_results.py.
The raw experimental results are saved in results/results.csv.
The generated figures used in the report are saved in results/figures/.
Instructions for running the code are provided in README.txt.

How to run:
1. Install dependencies: pip install -r requirements.txt
2. Run experiments: cd code && python main_experiments.py
3. Regenerate tables and figures from existing results: cd code && python main_experiments.py --aggregate-only

Quick test example:
cd code && python main_experiments.py --datasets mnist --models baseline --seeds 0 --epochs 1 --train-size 256 --val-size 128
