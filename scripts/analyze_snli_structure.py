import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter
import os

def analyze_snli_structure():
    """
    Loads the SNLI dataset, analyzes its structure for premise/hypothesis repetitions,
    and generates a report with statistics and visualizations.
    """
    # 1. Load the SNLI dataset from local pyarrow file
    print("Loading SNLI dataset from local arrow file...")
    # The original dataset has train, validation, and test splits. 
    # We are loading a single file, so we'll treat it as the 'train' split.
    snli_dataset = load_dataset('arrow', data_files={'train': 'data/snli/filtered/snli_filtered/data-00000-of-00001.arrow'})
    
    # Process each split (in this case, only 'train')
    for split in snli_dataset.keys():
        print(f"\nAnalyzing '{split}' split...")
        df = snli_dataset[split].to_pandas()

        # Clean data: remove entries with label -1 (unlabeled)
        df = df[df['label'] != -1].copy()
        if df.empty:
            print(f"No labeled data to analyze in '{split}' split.")
            continue

        # 2. Extract premise, hypothesis, and label columns
        premises = df['premise']
        hypotheses = df['hypothesis']
        labels = df['label']
        
        # Map labels to human-readable names
        label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        df['label_name'] = df['label'].map(label_map)

        # 3. Check for uniqueness
        unique_premises = set(premises)
        unique_hypotheses = set(hypotheses)

        # Do the same premises appear with different labels?
        premise_label_groups = df.groupby('premise')['label_name'].nunique()
        multi_label_premises = premise_label_groups[premise_label_groups > 1]

        # Do the same hypotheses appear with different labels?
        hypothesis_label_groups = df.groupby('hypothesis')['label_name'].nunique()
        multi_label_hypotheses = hypothesis_label_groups[hypothesis_label_groups > 1]
        
        # Check for duplicated (premise, hypothesis) pairs, ignoring the label
        duplicated_pairs_mask = df.duplicated(subset=['premise', 'hypothesis'], keep=False)
        num_duplicated_pair_rows = int(duplicated_pairs_mask.sum())

        # Check for duplicated pairs with different labels (potential errors)
        pair_label_groups = df.groupby(['premise', 'hypothesis'])['label_name'].nunique()
        conflicting_label_pairs = pair_label_groups[pair_label_groups > 1]
        num_conflicting_pairs = len(conflicting_label_pairs)
        
        print(f"Found {num_duplicated_pair_rows} rows that are part of a duplicated (premise, hypothesis) pair.")
        print(f"Found {num_conflicting_pairs} unique pairs with conflicting labels.")
        
        # Overlap between premise and hypothesis texts
        premise_hypothesis_overlap = unique_premises.intersection(unique_hypotheses)

        # 4. Generate summary statistics
        total_premises = len(premises)
        total_hypotheses = len(hypotheses)

        stats = {
            "Total premise instances": total_premises,
            "Total unique premises": len(unique_premises),
            "Total hypothesis instances": total_hypotheses,
            "Total unique hypotheses": len(unique_hypotheses),
            "Rows in duplicated (premise, hypothesis) pairs": num_duplicated_pair_rows,
            "Unique pairs with conflicting labels": num_conflicting_pairs,
            "Premises appearing with multiple labels": len(multi_label_premises),
            "Hypotheses appearing with multiple labels": len(multi_label_hypotheses),
            "Texts appearing as both premise and hypothesis": len(premise_hypothesis_overlap),
        }

        # Cross-contamination matrix
        is_premise = {text: True for text in unique_premises}
        is_hypothesis = {text: True for text in unique_hypotheses}
        all_texts = unique_premises.union(unique_hypotheses)
        
        contamination_counts = Counter()
        for text in all_texts:
            role = ("premise" if text in is_premise else "not_premise",
                    "hypothesis" if text in is_hypothesis else "not_hypothesis")
            contamination_counts[role] += 1
            
        contamination_matrix = pd.DataFrame(0, index=['is_premise', 'not_premise'], columns=['is_hypothesis', 'not_hypothesis'])
        for (p_role, h_role), count in contamination_counts.items():
            contamination_matrix.loc[p_role, h_role] = count
        
        # 5. Create visualizations
        output_dir = f"reports/snli_analysis/filtered/{split}"
        os.makedirs(output_dir, exist_ok=True)

        # Heatmap for cross-contamination
        plt.figure(figsize=(8, 6))
        sns.heatmap(contamination_matrix, annot=True, fmt='.0f', cmap='viridis')
        plt.title(f'Cross-Contamination of Texts in SNLI ({split})')
        plt.xlabel('Role')
        plt.ylabel('Role')
        plt.savefig(f"{output_dir}/cross_contamination_matrix.png")
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")

        # 6. Export a detailed report
        report_path = f"{output_dir}/detailed_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"SNLI Dataset Structure Analysis - '{split}' split\n")
            f.write("="*50 + "\n\n")
            
            f.write("Summary Statistics:\n")
            for key, value in stats.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")

            f.write("Cross-Contamination Matrix:\n")
            f.write(contamination_matrix.to_string() + "\n\n")

            f.write("="*50 + "\n")
            f.write("Examples of Texts with Multiple Roles/Labels\n")
            f.write("="*50 + "\n\n")

            # Examples of premises with multiple labels
            f.write("Premises with multiple labels (first 5 examples):\n")
            for premise_text, _ in multi_label_premises.head(5).items():
                f.write(f"- Premise: \"{premise_text}\"\n")
                examples = df[df['premise'] == premise_text]
                for _, row in examples.iterrows():
                    f.write(f"  - Hypothesis: \"{row['hypothesis']}\" -> Label: {row['label_name']}\n")
                f.write("\n")

            # Examples of pairs with conflicting labels
            f.write("Pairs with conflicting labels (first 5 examples):\n")
            if num_conflicting_pairs > 0:
                for (premise, hypothesis), _ in conflicting_label_pairs.head(5).items():
                    f.write(f"- Premise: \"{premise}\"\n")
                    f.write(f"  Hypothesis: \"{hypothesis}\"\n")
                    examples = df[(df['premise'] == premise) & (df['hypothesis'] == hypothesis)]
                    for _, row in examples.iterrows():
                        f.write(f"  -> Found Label: {row['label_name']}\n")
                    f.write("\n")
            else:
                f.write("None found.\n\n")

            # Examples of texts in both roles
            f.write("Texts appearing as both premise and hypothesis (first 5 examples):\n")
            for i, text in enumerate(list(premise_hypothesis_overlap)[:5]):
                f.write(f"- Text: \"{text}\"\n")
                premise_examples = df[df['premise'] == text]
                hypothesis_examples = df[df['hypothesis'] == text]
                if not premise_examples.empty:
                    f.write(f"  - As Premise: {len(premise_examples)} time(s)\n")
                if not hypothesis_examples.empty:
                    f.write(f"  - As Hypothesis: {len(hypothesis_examples)} time(s)\n")
                f.write("\n")
        
        print(f"Detailed report saved to {report_path}")

if __name__ == '__main__':
    analyze_snli_structure() 