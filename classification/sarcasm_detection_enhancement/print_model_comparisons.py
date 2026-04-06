import os
import pandas as pd
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
EXCEL_OUTPUT_PATH = os.path.join(RESULTS_DIR, 'evaluation_results.xlsx')
TEST_RESULTS_PATH = os.path.join(RESULTS_DIR, 'test_dataset_results.csv')
COMPARISON_COLUMNS = [
    'Model',
    'Accuracy',
    'Precision (macro)',
    'Recall (macro)',
    'F1-score (macro)',
    'F1-score (weighted)',
    'Throughput (records/sec)',
]
TITLE_LABELS = {
    'preprocessed_text': 'Preprocessed Text',
    'ner_text': 'NER Enhanced Text',
    'hybrid_text': 'Hybrid Text (Preprocessed + NER)',
    'augmented_text': 'Augmented Text (Preprocessed + NER)',
}

TABLE_THEMES = {
    'preprocessed_text': {
        'header_bg': '#264653',
        'header_fg': '#FFFFFF',
        'best_row_bg': '#C8F7C5',
        'stripe_even_bg': '#F7F7F7',
        'stripe_odd_bg': '#EBF2F7',
    },
    'augmented_text': {
        'header_bg': '#7F1D1D',
        'header_fg': '#FFF8E7',
        'best_row_bg': '#CDECCF',
        'stripe_even_bg': '#FFF7ED',
        'stripe_odd_bg': '#FFE4E6',
    },
    'default': {
        'header_bg': '#1F2937',
        'header_fg': '#FFFFFF',
        'best_row_bg': '#D1FAE5',
        'stripe_even_bg': '#F9FAFB',
        'stripe_odd_bg': '#F3F4F6',
    },
}


def load_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = pd.read_excel(EXCEL_OUTPUT_PATH, sheet_name='summary')
    detailed_df = pd.read_excel(EXCEL_OUTPUT_PATH, sheet_name='detailed_reports')
    return summary_df, detailed_df


def load_timed_record_count() -> int:
    if os.path.exists(TEST_RESULTS_PATH):
        test_df = pd.read_csv(TEST_RESULTS_PATH)
        return len(test_df)
    return 200


def extract_macro_metrics(detailed_df: pd.DataFrame, text_source: str, model: str) -> Dict[str, float]:
    subset = detailed_df[
        (detailed_df['text_source'] == text_source) &
        (detailed_df['model'] == model) &
        (detailed_df['label'] == 'macro avg')
    ]
    if subset.empty:
        return {'precision': 0, 'recall': 0, 'f1-score': 0}
    row = subset.iloc[0]
    return {
        'precision': round(float(row['precision']), 3),
        'recall': round(float(row['recall']), 3),
        'f1-score': round(float(row['f1-score']), 3),
    }


def build_comparison_table(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    text_source: str,
    timed_record_count: int,
) -> pd.DataFrame:
    source_summary = summary_df[summary_df['text_source'] == text_source].copy()
    source_summary = source_summary.sort_values('weighted_f1', ascending=False).reset_index(drop=True)
    comparison_data = []
    for _, row in source_summary.iterrows():
        model = row['model']
        macro_metrics = extract_macro_metrics(detailed_df, text_source, model)
        time_sec = float(row['classification_time_seconds'])
        throughput = timed_record_count / time_sec if time_sec > 0 else float('inf')

        comparison_data.append({
            'Model': model,
            'Accuracy': round(float(row['accuracy']), 3),
            'Precision (macro)': macro_metrics['precision'],
            'Recall (macro)': macro_metrics['recall'],
            'F1-score (macro)': macro_metrics['f1-score'],
            'F1-score (weighted)': round(float(row['weighted_f1']), 3),
            'Throughput (records/sec)': round(throughput, 2) if throughput != float('inf') else 'inf',
        })
    
    return pd.DataFrame(comparison_data, columns=COMPARISON_COLUMNS)


def resolve_text_sources(summary_df: pd.DataFrame) -> List[Tuple[str, str]]:
    preferred_order = ['preprocessed_text', 'ner_text', 'hybrid_text', 'augmented_text']
    available = set(summary_df['text_source'].dropna().astype(str).tolist())

    resolved = [
        (source, TITLE_LABELS.get(source, source.replace('_', ' ').upper()))
        for source in preferred_order
        if source in available
    ]

    for source in sorted(available):
        if source not in preferred_order:
            resolved.append((source, TITLE_LABELS.get(source, source.replace('_', ' ').upper())))

    return resolved


def print_table_with_formatting(title: str, df: pd.DataFrame, highlight_index: int = 0):
    print(f"\n{'='*120}")
    print(f"{title:^120}")
    print(f"{'='*120}")
    header = " | ".join(f"{col:^20}" for col in df.columns)
    print(header)
    print('-' * 120)
    for idx, row in df.iterrows():
        values = [str(row[col]) for col in df.columns]
        
        line = " | ".join(f"{val:^20}" for val in values)
        
        if idx == highlight_index:
            print(f">>> {line} <<<  ⭐ BEST MODEL")
        else:
            print(line)
    
    print(f"{'='*120}\n")


def create_comparison_image(summary_df: pd.DataFrame, detailed_df: pd.DataFrame, text_source: str, output_path: str):
    timed_record_count = load_timed_record_count()
    comparison_df = build_comparison_table(summary_df, detailed_df, text_source, timed_record_count)
    if comparison_df.empty:
        print(f"- Skipping image for {text_source}: no rows found.")
        return

    display_df = comparison_df.drop(columns=['Throughput (records/sec)'], errors='ignore').copy()
    theme = TABLE_THEMES.get(text_source, TABLE_THEMES['default'])
    
    fig_height = max(5, 0.8 * (len(display_df) + 2))
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis('off')
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc='center',
        cellLoc='center',
        colLoc='center',
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color=theme['header_fg'], fontsize=11)
            cell.set_facecolor(theme['header_bg'])
        else:
            data_row = row - 1
            if data_row == 0:
                cell.set_facecolor(theme['best_row_bg'])
                if col == 0:
                    cell.set_text_props(weight='bold', fontsize=11)
            elif data_row % 2 == 0:
                cell.set_facecolor(theme['stripe_even_bg'])
            else:
                cell.set_facecolor(theme['stripe_odd_bg'])
    
    plt.title(
        f"Model Performance Comparison - {TITLE_LABELS.get(text_source, text_source)}",
        fontsize=16,
        weight='bold',
        pad=20
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Comparison image saved: {output_path}")


def main():
    print("\n" + "="*120)
    print("MODEL EVALUATION RESULTS AND PERFORMANCE COMPARISON".center(120))
    print("="*120)
    
    summary_df, detailed_df = load_results()
    timed_record_count = load_timed_record_count()
    
    text_sources = resolve_text_sources(summary_df)
    if not text_sources:
        print("No text sources found in summary data. Nothing to compare.")
        return

    for text_source, title in text_sources:
        comparison_df = build_comparison_table(summary_df, detailed_df, text_source, timed_record_count)
        if comparison_df.empty:
            print(f"\n{'='*120}")
            print(f"{title:^120}")
            print(f"{'='*120}")
            print("No rows found for this text source.")
            print(f"{'='*120}\n")
            continue
        print_table_with_formatting(title.upper(), comparison_df)

    print("\nGenerating visual comparison tables...")

    for text_source, _ in text_sources:
        image_name = f"model_comparison_detailed_{text_source}.png"
        image_path = os.path.join(RESULTS_DIR, image_name)
        create_comparison_image(summary_df, detailed_df, text_source, image_path)

    print("\n" + "="*120)
    print("KEY FINDINGS".center(120))
    print("="*120)
    for text_source, title in text_sources:
        source_rows = summary_df[summary_df['text_source'] == text_source].copy()
        if source_rows.empty:
            continue
        best = source_rows.sort_values('weighted_f1', ascending=False).iloc[0]
        print(f"\n{title}:")
        print(f"  • Best Model: {best['model']}")
        print(f"  • F1-score (weighted): {best['weighted_f1']:.3f}")
        print(f"  • Accuracy: {best['accuracy']:.3f}")
        throughput = timed_record_count / best['classification_time_seconds'] if best['classification_time_seconds'] > 0 else float('inf')
        print(f"  • Throughput: {throughput:.2f} records/sec")

    overall_best = summary_df.sort_values('weighted_f1', ascending=False).iloc[0]
    print(f"\n{'Overall Best Configuration:':^120}")
    print(f"  • Model: {overall_best['model']} with {overall_best['text_source']}")
    print(f"  • F1-score: {overall_best['weighted_f1']:.3f}")
    print(f"  • Accuracy: {overall_best['accuracy']:.3f}")
    
    print("\n" + "="*120)
    print("✓ All comparisons completed. Check the results/ folder for visual comparison images.".center(120))
    print("="*120 + "\n")


if __name__ == '__main__':
    main()
