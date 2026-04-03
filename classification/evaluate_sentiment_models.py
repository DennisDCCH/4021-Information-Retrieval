from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_PATH = os.path.join(BASE_DIR, 'evaluation', 'evaluation_dataset.xlsx')
PREPROCESSED_PATH = os.path.join(BASE_DIR, '..', 'preprocessing', 'preprocessed_data', 'preprocessed_data.csv')
NER_PATH = os.path.join(BASE_DIR, '..', 'preprocessing', 'NER_preprocessed_data', 'NER_preprocessed_data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
EXCEL_OUTPUT_PATH = os.path.join(RESULTS_DIR, 'evaluation_results.xlsx')
COMPARISON_IMAGE_PREPROCESSED_PATH = os.path.join(RESULTS_DIR, 'model_comparison_preprocessed_text.png')
COMPARISON_IMAGE_NER_PATH = os.path.join(RESULTS_DIR, 'model_comparison_ner_text.png')
COMPARISON_IMAGE_HYBRID_PATH = os.path.join(RESULTS_DIR, 'model_comparison_hybrid_text.png')

os.makedirs(RESULTS_DIR, exist_ok=True)

TRANSFORMER_MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128
RANDOM_STATE = 99
TRANSFORMER_VAL_SIZE = 0.15


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    factory: Optional[Callable[[], object]] = None


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec('Logistic Regression', 'classical', lambda: LogisticRegression(max_iter=200, class_weight='balanced')),
    ModelSpec('SVM', 'classical', lambda: SVC(kernel='linear', class_weight='balanced', probability=True, random_state=RANDOM_STATE)),
    ModelSpec('DummyClassifier', 'classical', lambda: DummyClassifier(strategy='uniform', random_state=RANDOM_STATE)),
    ModelSpec('Transformer (DistilBERT)', 'transformer'),
]

TEXT_SOURCES: List[Tuple[str, str]] = [
    ('preprocessed_text', PREPROCESSED_PATH),
    ('ner_text', NER_PATH),
    ('hybrid_text', ''),
]


class EncodedTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer: AutoTokenizer):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def ensure_review_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'review_id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'review_id'})
    return df


def load_evaluation_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = pd.read_excel(EVAL_PATH)
    eval_df = ensure_review_id(eval_df)

    eval_df['ground_truth_polarity'] = (
        eval_df['ground_truth_polarity']
        .astype(str)
        .str.strip()
        .str.upper()
    )

    eval_df['polarity'] = eval_df['ground_truth_polarity'].map({
        'POSITIVE': 1,
        'NEGATIVE': 0,
        'NEUTRAL': 2,
    })

    df_pol = eval_df[eval_df['polarity'] != 2].copy()

    train_meta, test_meta = train_test_split(
        df_pol[['review_id', 'ground_truth_polarity', 'polarity']],
        test_size=0.2,
        stratify=df_pol['polarity'],
        random_state=RANDOM_STATE,
    )

    return train_meta.copy(), test_meta.copy()


def load_text_source(path: str, output_column: str) -> pd.DataFrame:
    source_df = pd.read_csv(path, usecols=['review_id', 'processed_text'])
    source_df = ensure_review_id(source_df)
    source_df = source_df.rename(columns={'processed_text': output_column})
    return source_df[['review_id', output_column]].copy()


def build_shared_datasets(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    preprocessed_df = load_text_source(PREPROCESSED_PATH, 'preprocessed_text')
    ner_df = load_text_source(NER_PATH, 'ner_text')

    train_df = (
        train_meta
        .merge(preprocessed_df, on='review_id', how='left')
        .merge(ner_df, on='review_id', how='left')
    )
    test_df = (
        test_meta
        .merge(preprocessed_df, on='review_id', how='left')
        .merge(ner_df, on='review_id', how='left')
    )


    train_df['ner_text'] = train_df['ner_text'].fillna('')
    test_df['ner_text'] = test_df['ner_text'].fillna('')
    train_df['hybrid_text'] = (train_df['preprocessed_text'].fillna('') + ' ' + train_df['ner_text']).str.strip()
    test_df['hybrid_text'] = (test_df['preprocessed_text'].fillna('') + ' ' + test_df['ner_text']).str.strip()

    missing_train = train_df['preprocessed_text'].isna().sum()
    missing_test = test_df['preprocessed_text'].isna().sum()
    if missing_train or missing_test:
        raise ValueError(
            f'Missing preprocessed_text values: train={missing_train}, test={missing_test}. '
            'Check that review_id matches across the files.'
        )

    return train_df, test_df


def classical_model_report(
    model_name: str,
    model_factory: Callable[[], object],
    x_train_vec,
    y_train: pd.Series,
    x_test_vec,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.Series]:
    model = model_factory()
    model.fit(x_train_vec, y_train)

    start_time = time.perf_counter()
    y_pred = model.predict(x_test_vec)
    classification_time = time.perf_counter() - start_time

    report = classification_report(
        y_test,
        y_pred,
        digits=3,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'label'})
    report_df['label'] = report_df['label'].replace({
        '0': 'NEGATIVE',
        '1': 'POSITIVE',
        0: 'NEGATIVE',
        1: 'POSITIVE',
    })
    summary_row = {
        'model': model_name,
        'accuracy': report.get('accuracy', 0.0),
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'classification_time_seconds': classification_time,
    }
    predictions = pd.Series(y_pred, index=y_test.index)
    return report_df, summary_row, predictions


def transformer_model_report(
    model_name: str,
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.Series]:
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    stratify_labels = y_train if y_train.nunique() > 1 and y_train.value_counts().min() >= 2 else None
    x_train_fit, x_val, y_train_fit, y_val = train_test_split(
        x_train.fillna(''),
        y_train,
        test_size=TRANSFORMER_VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    train_dataset = EncodedTextDataset(x_train_fit, y_train_fit, tokenizer)
    val_dataset = EncodedTextDataset(x_val, y_val, tokenizer)
    test_dataset = EncodedTextDataset(x_test.fillna(''), y_test, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, 'transformer_runs', model_name.replace(' ', '_').lower()),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=[],
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
        disable_tqdm=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    start_time = time.perf_counter()
    prediction_output = trainer.predict(test_dataset)
    classification_time = time.perf_counter() - start_time

    y_pred = prediction_output.predictions.argmax(axis=1)
    report = classification_report(
        y_test,
        y_pred,
        digits=3,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'label'})
    report_df['label'] = report_df['label'].replace({
        '0': 'NEGATIVE',
        '1': 'POSITIVE',
        0: 'NEGATIVE',
        1: 'POSITIVE',
    })
    summary_row = {
        'model': model_name,
        'accuracy': report.get('accuracy', 0.0),
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'classification_time_seconds': classification_time,
    }
    predictions = pd.Series(y_pred, index=y_test.index)
    return report_df, summary_row, predictions


def evaluate_text_source(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
    x_train = train_df[text_column].fillna('')
    y_train = train_df['polarity']
    x_test = test_df[text_column].fillna('')
    y_test = test_df['polarity']

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    detailed_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    predictions: Dict[str, pd.Series] = {}

    for spec in MODEL_SPECS:
        if spec.kind == 'classical':
            if spec.factory is None:
                raise ValueError(f'Missing factory for classical model {spec.name}')
            report_df, summary_row, y_pred = classical_model_report(
                spec.name,
                spec.factory,
                x_train_vec,
                y_train,
                x_test_vec,
                y_test,
            )
        elif spec.kind == 'transformer':
            report_df, summary_row, y_pred = transformer_model_report(
                spec.name,
                x_train,
                y_train,
                x_test,
                y_test,
            )
        else:
            raise ValueError(f'Unknown model kind: {spec.kind}')

        report_df.insert(0, 'text_source', text_column)
        report_df.insert(1, 'model', spec.name)
        detailed_rows.append(report_df)

        summary_row['text_source'] = text_column
        summary_rows.append(summary_row)

        predictions[spec.name] = y_pred

    detailed_report_df = pd.concat(detailed_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return detailed_report_df, summary_df, predictions


def choose_best_model(summary_df: pd.DataFrame, text_source: str) -> str:
    source_summary = summary_df[summary_df['text_source'] == text_source].copy()
    best_row = source_summary.sort_values(['weighted_f1', 'accuracy'], ascending=False).iloc[0]
    return str(best_row['model'])


def save_comparison_table_image(summary_df: pd.DataFrame, text_source: str, output_path: str) -> None:
    display_df = summary_df[summary_df['text_source'] == text_source].copy()
    best_model = choose_best_model(summary_df, text_source)
    display_df['selected_model'] = display_df['model'] == best_model

    display_df = display_df.sort_values(['weighted_f1', 'accuracy'], ascending=[False, False]).reset_index(drop=True)
    display_df['classification_time_seconds'] = display_df['classification_time_seconds'].map(lambda value: f'{value:.4f}')
    display_df['accuracy'] = display_df['accuracy'].map(lambda value: f'{value:.3f}')
    display_df['macro_f1'] = display_df['macro_f1'].map(lambda value: f'{value:.3f}')
    display_df['weighted_f1'] = display_df['weighted_f1'].map(lambda value: f'{value:.3f}')

    table_df = display_df[['model', 'accuracy', 'macro_f1', 'weighted_f1', 'classification_time_seconds']].copy()

    fig_height = max(4.5, 0.7 * (len(table_df) + 2))
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc='center',
        cellLoc='center',
        colLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    selected_rows = display_df.index[display_df['selected_model']].tolist()
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#264653')
        else:
            data_row = row - 1
            if data_row in selected_rows:
                cell.set_facecolor('#c8f7c5')
                if col == 1:
                    cell.set_text_props(weight='bold')
            elif data_row % 2 == 0:
                cell.set_facecolor('#f7f7f7')
            else:
                cell.set_facecolor('#ebf2f7')

    title_label_map = {
        'preprocessed_text': 'preprocessed_text',
        'ner_text': 'ner_text',
        'hybrid_text': 'hybrid_text (preprocessed_text + ner_text)',
    }
    title_text = f"Model Comparison Summary - {title_label_map.get(text_source, text_source)}"
    plt.title(title_text, fontsize=18, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    train_meta, test_meta = load_evaluation_split()
    train_df, test_df = build_shared_datasets(train_meta, test_meta)

    test_dataset = test_df[['review_id', 'ground_truth_polarity', 'preprocessed_text', 'ner_text', 'hybrid_text']].copy()

    comparison_tables: List[pd.DataFrame] = []
    summary_tables: List[pd.DataFrame] = []
    prediction_store: Dict[Tuple[str, str], pd.Series] = {}

    for text_source, _ in TEXT_SOURCES:
        detailed_report_df, summary_df, predictions = evaluate_text_source(train_df, test_df, text_source)
        comparison_tables.append(detailed_report_df)
        summary_tables.append(summary_df)

        for model_name, series in predictions.items():
            prediction_store[(text_source, model_name)] = series

    comparison_df = pd.concat(comparison_tables, ignore_index=True)
    summary_df = pd.concat(summary_tables, ignore_index=True)

    best_preprocessed_model = choose_best_model(summary_df, 'preprocessed_text')
    best_ner_model = choose_best_model(summary_df, 'ner_text')
    best_hybrid_model = choose_best_model(summary_df, 'hybrid_text')

    test_dataset['preprocessed_text_prediction_model'] = best_preprocessed_model
    test_dataset['preprocessed_text_prediction_result'] = (
        prediction_store[('preprocessed_text', best_preprocessed_model)]
        .map({1: 'POSITIVE', 0: 'NEGATIVE'})
    )
    test_dataset['ner_text_prediction_model'] = best_ner_model
    test_dataset['ner_text_prediction_result'] = (
        prediction_store[('ner_text', best_ner_model)]
        .map({1: 'POSITIVE', 0: 'NEGATIVE'})
    )
    test_dataset['hybrid_text_prediction_model'] = best_hybrid_model
    test_dataset['hybrid_text_prediction_result'] = (
        prediction_store[('hybrid_text', best_hybrid_model)]
        .map({1: 'POSITIVE', 0: 'NEGATIVE'})
    )

    test_dataset_path = os.path.join(RESULTS_DIR, 'test_dataset_results.csv')
    test_dataset.to_csv(test_dataset_path, index=False)

    comparison_csv_path = os.path.join(RESULTS_DIR, 'comparison_table.csv')
    summary_csv_path = os.path.join(RESULTS_DIR, 'comparison_summary.csv')
    comparison_df.to_csv(comparison_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    save_comparison_table_image(summary_df, 'preprocessed_text', COMPARISON_IMAGE_PREPROCESSED_PATH)
    save_comparison_table_image(summary_df, 'ner_text', COMPARISON_IMAGE_NER_PATH)
    save_comparison_table_image(summary_df, 'hybrid_text', COMPARISON_IMAGE_HYBRID_PATH)

    with pd.ExcelWriter(EXCEL_OUTPUT_PATH) as writer:
        summary_df.sort_values(['text_source', 'weighted_f1', 'accuracy'], ascending=[True, False, False]).to_excel(
            writer,
            sheet_name='summary',
            index=False,
        )
        comparison_df.to_excel(writer, sheet_name='detailed_reports', index=False)
        test_dataset.to_excel(writer, sheet_name='test_dataset_results', index=False)

    print('Evaluation split size:', len(test_dataset))
    print('Best preprocessed_text model:', best_preprocessed_model)
    print('Best ner_text model:', best_ner_model)
    print('Best hybrid_text model:', best_hybrid_model)
    print('\nComparison table saved to:', comparison_csv_path)
    print('Summary table saved to:', summary_csv_path)
    print('Comparison image saved to:', COMPARISON_IMAGE_PREPROCESSED_PATH)
    print('Comparison image saved to:', COMPARISON_IMAGE_NER_PATH)
    print('Comparison image saved to:', COMPARISON_IMAGE_HYBRID_PATH)
    print('Excel workbook saved to:', EXCEL_OUTPUT_PATH)
    print('Test dataset results saved to:', test_dataset_path)


if __name__ == '__main__':
    main()