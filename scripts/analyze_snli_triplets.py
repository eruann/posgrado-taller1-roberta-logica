import pandas as pd
from datasets import load_from_disk
import os
from collections import Counter

def analyze_snli_triplets():
    """
    Analyzes the filtered SNLI dataset to determine if premises systematically
    appear with hypotheses for all three labels (entailment, contradiction, neutral).
    This script verifies the "systematic triplet" assumption for contrastive analysis.
    """
    # 1. Load the filtered SNLI dataset
    print("Loading filtered SNLI dataset...")
    try:
        # Path after running the filter script
        dataset_path = 'data/snli/dataset/snli_filtered'
        ds = load_from_disk(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure you have run the 'filter_snli_cross_contamination.py' script first.")
        return

    df = ds.to_pandas()
    print(f"Dataset loaded with {len(df)} records.")

    # Map labels to human-readable names
    label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    df['label_name'] = df['label'].map(label_map)

    # 2. Group by premise and count unique labels
    print("Analyzing premise-label structure...")
    premise_label_counts = df.groupby('premise')['label_name'].nunique()
    total_unique_premises = len(premise_label_counts)

    # Count premises by the number of unique labels they have
    label_distribution = premise_label_counts.value_counts().sort_index()
    complete_triplets_count = label_distribution.get(3, 0)
    doubles_count = label_distribution.get(2, 0)
    singles_count = label_distribution.get(1, 0)
    
    # 3. Analyze the structure of complete triplets
    if complete_triplets_count > 0:
        triplet_premises = premise_label_counts[premise_label_counts == 3].index
        triplet_df = df[df['premise'].isin(triplet_premises)]
        
        # Count how many hypotheses per label for each triplet premise
        hypotheses_per_label = triplet_df.groupby(['premise', 'label_name']).size().unstack(fill_value=0)
        
        # Check for balance (e.g., how many are perfectly balanced like 1-1-1)
        balanced_triplets = hypotheses_per_label[
            (hypotheses_per_label['entailment'] == hypotheses_per_label['contradiction']) &
            (hypotheses_per_label['entailment'] == hypotheses_per_label['neutral'])
        ]
        num_perfectly_balanced = len(balanced_triplets)
        
        # Calculate average hypotheses per label for these complete triplets
        avg_hypotheses_per_label = hypotheses_per_label.mean()
    else:
        hypotheses_per_label = None
        num_perfectly_balanced = 0
        avg_hypotheses_per_label = pd.Series([0,0,0], index=['entailment', 'neutral', 'contradiction'])

    # 4. Generate statistics and report
    output_dir = "reports/snli_analysis"
    os.makedirs(output_dir, exist_ok=True)
    report_path = f"{output_dir}/triplet_analysis_report.txt"
    
    print(f"Generating report at {report_path}...")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SNLI Dataset Triplet Structure Analysis\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total unique premises found: {total_unique_premises}\n\n")

        f.write("Premise Distribution by Number of Associated Labels:\n")
        f.write(f"- Premises with 1 label (not contrastive): {singles_count} ({singles_count / total_unique_premises:.2%})\n")
        f.write(f"- Premises with 2 labels (partially contrastive): {doubles_count} ({doubles_count / total_unique_premises:.2%})\n")
        f.write(f"- Premises with 3 labels (complete triplets): {complete_triplets_count} ({complete_triplets_count / total_unique_premises:.2%})\n\n")

        f.write("Analysis of Complete Triplets:\n")
        f.write(f"- Total complete triplets found: {complete_triplets_count}\n")
        if total_unique_premises > 0:
            f.write(f"- Percentage of premises forming complete triplets: {complete_triplets_count / total_unique_premises:.2%}\n")
        f.write(f"- Number of perfectly balanced triplets (e.g., 1-1-1 or 2-2-2 hypotheses): {num_perfectly_balanced}\n")
        f.write("- Average number of hypotheses per label within these triplets:\n")
        f.write(f"  - Entailment: {avg_hypotheses_per_label['entailment']:.2f}\n")
        f.write(f"  - Neutral: {avg_hypotheses_per_label['neutral']:.2f}\n")
        f.write(f"  - Contradiction: {avg_hypotheses_per_label['contradiction']:.2f}\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("5. Verification of the 'Systematic Triplet' Assumption\n")
        f.write("=" * 50 + "\n\n")

        percentage_contrastive = (doubles_count + complete_triplets_count) / total_unique_premises if total_unique_premises > 0 else 0
        f.write(f"The percentage of premises that allow for any contrastive analysis (having at least 2 different labels) is: {percentage_contrastive:.2%}\n")
        
        percentage_full_triplets = complete_triplets_count / total_unique_premises if total_unique_premises > 0 else 0
        f.write(f"The percentage of premises that form a complete triplet (all 3 labels) is: {percentage_full_triplets:.2%}\n\n")
        
        if percentage_full_triplets > 0.5:
            f.write("Conclusion: The 'systematic triplet' assumption appears to be LARGELY VALID.\n")
            f.write("A significant majority of premises are paired with hypotheses covering all three inference types, making the dataset highly suitable for the proposed contrastive analysis.\n")
        elif percentage_full_triplets > 0.1:
            f.write("Conclusion: The 'systematic triplet' assumption is PARTIALLY VALID.\n")
            f.write("While not a majority, a substantial number of premises form complete triplets. It is possible to perform the contrastive analysis, but the analysis will be limited to this specific subset of the data.\n")
        else:
            f.write("Conclusion: The 'systematic triplet' assumption appears to be INVALID.\n")
            f.write("A very small fraction of premises form complete triplets. Relying on this structure for a broad analysis is not feasible. The experimental design may need to be revised to focus on pairs (e.g., entailment vs. contradiction) rather than triplets.\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("Examples of Complete Triplets (Top 5)\n")
        f.write("=" * 50 + "\n\n")

        if complete_triplets_count > 0:
            for premise_text in hypotheses_per_label.head(5).index:
                f.write(f"Premise: \"{premise_text}\"\n")
                examples = df[df['premise'] == premise_text]
                for _, row in examples.iterrows():
                    f.write(f"  - ({row['label_name']}) Hypothesis: \"{row['hypothesis']}\"\n")
                f.write("\n")
        else:
            f.write("No complete triplets found to display.\n")

    print("Analysis complete. Report saved.")

if __name__ == '__main__':
    analyze_snli_triplets() 