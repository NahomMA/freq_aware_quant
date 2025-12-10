import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def analyze_dataset(file_path: str):
    """
    Analyzes the ConvBench dataset and generates a summary report.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Dataset not found at {path}")
        return

    # Load Dataset
    try:
        df = pd.read_excel(path)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 1. Basic Statistics
    num_samples = len(df)
    cols = df.columns.tolist()
    
    # Assume column names based on typical ConvBench structure or inspecting the first few rows
    # If columns aren't named standardly, we might need to guess. 
    # For now, let's look for columns that look like conversation turns or 'human'/'gpt'
    
    # Calculate Turn Counts and Lengths
    # Heuristic: Treat every non-empty cell in a row as a potential turn if we don't know schema
    # OR if there's a 'conversations' column (JSON)
    
    # Let's assume a simple structure for now or print columns to debug
    print(f"Columns: {cols}")
    
    # 2. Text Length Distribution
    # We will iterate through rows and count total characters per conversation
    lengths = []
    turn_counts = []
    
    for idx, row in df.iterrows():
        # Naive approach: concantenate all string cells
        full_text = " ".join([str(x) for x in row.values if isinstance(x, str)])
        lengths.append(len(full_text))
        
        # Naive turn count: simple split by likely separators or cell count
        # If the excel has "User", "Assistant" columns, use those.
        # For now, let's assume row width represents turns if wide format
        turn_counts.append(row.count()) 

    # 3. Visualization
    sns.set_style("whitegrid")
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Length Histogram
    sns.histplot(lengths, kde=True, ax=axs[0], color='skyblue')
    axs[0].set_title("Distribution of Conversation Lengths (Characters)")
    axs[0].set_xlabel("Character Count")
    
    # Turn Count Histogram
    sns.histplot(turn_counts, kde=False, bins=range(min(turn_counts), max(turn_counts)+2), ax=axs[1], color='salmon')
    axs[1].set_title("Distribution of Non-Empty Cells (Proxy for Turns)")
    axs[1].set_xlabel("Count")
    
    plt.tight_layout()
    plot_path = path.parent / "dataset_stats.png"
    plt.savefig(plot_path)
    print(f"Saved distribution plot to {plot_path}")

    # 4. Generate Report
    report_path = path.parent / "dataset_report.md"
    with open(report_path, "w") as f:
        f.write("# ConvBench Dataset Report\n\n")
        f.write(f"**File Analyzed:** `{path.name}`\n")
        f.write(f"**Total Samples:** {num_samples}\n\n")
        f.write("## Statistics\n")
        f.write(f"- **Average Length:** {sum(lengths)/len(lengths):.2f} chars\n")
        f.write(f"- **Min Length:** {min(lengths)}\n")
        f.write(f"- **Max Length:** {max(lengths)}\n")
        f.write(f"- **Average Turns (Est):** {sum(turn_counts)/len(turn_counts):.1f}\n\n")
        f.write("## Content Sample (First Row)\n")
        f.write("```\n")
        # Write a safe snippet of the first row
        first_row_text = " | ".join([str(x)[:50] for x in df.iloc[0].values])
        f.write(first_row_text + "...\n")
        f.write("```\n\n")
        f.write("## Visualizations\n")
        f.write(f"![Distribution](dataset_stats.png)\n")

    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    # Update this path if your xlsx is elsewhere
    # Use absolute path or correct relative path from where script is run
    dataset_path = "mileston-1/src/data/ConvBench.xlsx"
    analyze_dataset(dataset_path)

