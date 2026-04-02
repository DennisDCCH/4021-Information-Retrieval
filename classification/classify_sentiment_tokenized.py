from sklearn.dummy import DummyClassifier
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as npm
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_PATH = os.path.join(BASE_DIR, 'evaluation', 'evaluation_dataset.xlsx')
CORPUS_PATH = os.path.join(BASE_DIR, '..', 'preprocessing', 'Temp_preprocessed_data.csv')
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

eval_df = pd.read_excel(EVAL_PATH)

eval_df['text'] = (eval_df['summary'].fillna('') + ' ' + eval_df['content'].fillna('')).str.strip()

eval_df['ground_truth_polarity'] = (
    eval_df['ground_truth_polarity']
    .astype(str)
    .str.strip()
    .str.upper()
)

# encode labels
eval_df['polarity'] = eval_df['ground_truth_polarity'].map({
    'POSITIVE': 1,
    'NEGATIVE': 0,
    'NEUTRAL': 2
})

# filter only polarity labels (exclude neutral)
df_pol = eval_df[eval_df['polarity'] != 2]


print("Evaluation dataset size:", len(df_pol))
# show polarity distribution with clear labels
label_map = {1: 'POSITIVE', 0: 'NEGATIVE'}
gt_counts = df_pol['polarity'].map(label_map).value_counts()
print("Ground Truth Polarity Distribution:")
print(gt_counts.to_string())

X_pol = df_pol['text']
y_pol = df_pol['polarity']


X_train_pol, X_test_pol, y_train_pol, y_test_pol = train_test_split(
    X_pol, y_pol, test_size=0.2, stratify=y_pol, random_state=99
)

# save the 20% evaluation test dataset
eval_test_df = pd.DataFrame({'text': X_test_pol, 'polarity': y_test_pol})
eval_test_df.to_csv(os.path.join(BASE_DIR, 'results', 'evaluation_test_dataset_tokenized.csv'), index=False)


# DistilBERT for contextual embeddings
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_bert_embeddings(texts, tokenizer, model, batch_size=32, device='cpu'):
    model = model.to(device)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = list(texts[i:i+batch_size])
        encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pooling
            last_hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            all_embeddings.append(mean_pooled.cpu())
    return torch.cat(all_embeddings, dim=0).numpy()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
X_train_vec_pol = get_bert_embeddings(X_train_pol, tokenizer, bert_model, device=device)


# train Logistic Regression model
clf_pol = LogisticRegression(max_iter=200, class_weight='balanced')
clf_pol.fit(X_train_vec_pol, y_train_pol)

# train SVM model
clf_svm = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=99)
clf_svm.fit(X_train_vec_pol, y_train_pol)


# evaluate
X_test_vec_pol = get_bert_embeddings(X_test_pol, tokenizer, bert_model, device=device)

# logistic Regression report
ml_report = classification_report(y_test_pol, clf_pol.predict(X_test_vec_pol), digits=3, output_dict=True)
ml_report_df = pd.DataFrame(ml_report).transpose()
ml_report_df = ml_report_df.rename(index={'0': 'NEGATIVE', '1': 'POSITIVE'})
print("\nLogistic Regression Polarity Detection Report:")
print(ml_report_df.to_string(float_format="{:.2f}".format))

# SVM report
svm_report = classification_report(y_test_pol, clf_svm.predict(X_test_vec_pol), digits=3, output_dict=True)
svm_report_df = pd.DataFrame(svm_report).transpose()
svm_report_df = svm_report_df.rename(index={'0': 'NEGATIVE', '1': 'POSITIVE'})
print("\nSVM Polarity Detection Report:")
print(svm_report_df.to_string(float_format="{:.2f}".format))

# dummyClassifier random baseline report
dummy_clf = DummyClassifier(strategy='uniform', random_state=99)
dummy_clf.fit(X_train_vec_pol, y_train_pol)
dummy_preds = dummy_clf.predict(X_test_vec_pol)
dummy_report = classification_report(y_test_pol, dummy_preds, digits=3, output_dict=True)
dummy_report_df = pd.DataFrame(dummy_report).transpose()
dummy_report_df = dummy_report_df.rename(index={'0': 'NEGATIVE', '1': 'POSITIVE'})
print("\nDummyClassifier (Random) Baseline Report:")
print(dummy_report_df.to_string(float_format="{:.2f}".format))

# save DummyClassifier report as styled table image
fig2, ax2 = plt.subplots(figsize=(10, 3.5))
ax2.axis('off')
tbl2 = ax2.table(
    cellText=dummy_report_df.round(2).values,
    colLabels=dummy_report_df.columns,
    rowLabels=dummy_report_df.index,
    loc='center',
    cellLoc='center',
    colColours=['#7a3b2e']*len(dummy_report_df.columns),
    rowColours=['#7a3b2e']*len(dummy_report_df.index)
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(12)
tbl2.scale(1.3, 1.3)
for (row, col), cell in tbl2.get_celld().items():
    if row == 0 or col == -1:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#7a3b2e')
    elif row % 2 == 0:
        cell.set_facecolor('#f7e6e0')
    else:
        cell.set_facecolor('#f2d6c2')
tbl2.auto_set_column_width(col=list(range(len(dummy_report_df.columns))))
plt.title('DummyClassifier (Random) Baseline Report', fontsize=16, pad=18, weight='bold', color='#7a3b2e')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results', 'dummy_classifier_report_tokenizer.png'), bbox_inches='tight', dpi=200)
plt.close(fig2)

metrics = ['precision', 'recall', 'f1-score']
labels = ['NEGATIVE', 'POSITIVE']
ml_scores = ml_report_df.loc[labels, metrics].astype(float)
svm_scores = svm_report_df.loc[labels, metrics].astype(float)
dummy_scores = dummy_report_df.loc[labels, metrics].astype(float)

fig3, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
ml_scores.plot(kind='bar', ax=axes[0], title='Logistic Regression', color=['#2d415b', '#4f81bd', '#a6bddb'])
svm_scores.plot(kind='bar', ax=axes[1], title='SVM', color=['#3b7a57', '#63c97b', '#c2f2d6'])
dummy_scores.plot(kind='bar', ax=axes[2], title='DummyClassifier', color=['#7a3b2e', '#c97b63', '#f2d6c2'])

for ax in axes:
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
fig3.suptitle('Comparison: Logistic Regression vs. SVM vs. DummyClassifier', fontsize=18, weight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(BASE_DIR, 'results', 'ml_vs_svm_vs_dummy_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(fig3)

# save SVM report as styled table image
fig_svm, ax_svm = plt.subplots(figsize=(10, 3.5))
ax_svm.axis('off')
tbl_svm = ax_svm.table(
    cellText=svm_report_df.round(2).values,
    colLabels=svm_report_df.columns,
    rowLabels=svm_report_df.index,
    loc='center',
    cellLoc='center',
    colColours=['#3b7a57']*len(svm_report_df.columns),
    rowColours=['#3b7a57']*len(svm_report_df.index)
)
tbl_svm.auto_set_font_size(False)
tbl_svm.set_fontsize(12)
tbl_svm.scale(1.3, 1.3)
for (row, col), cell in tbl_svm.get_celld().items():
    if row == 0 or col == -1:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#3b7a57')
    elif row % 2 == 0:
        cell.set_facecolor('#e0f7e6')
    else:
        cell.set_facecolor('#c2f2d6')
tbl_svm.auto_set_column_width(col=list(range(len(svm_report_df.columns))))
plt.title('SVM Polarity Detection Report', fontsize=16, pad=18, weight='bold', color='#3b7a57')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results', 'svm_polarity_detection_report_tokenizer.png'), bbox_inches='tight', dpi=200)
plt.close(fig_svm)

print("\nComparison of Logistic Regression, SVM, and DummyClassifier (Random Baseline):")
print("Logistic Regression Accuracy: {:.2f}".format(ml_report['accuracy']))
print("SVM Accuracy: {:.2f}".format(svm_report['accuracy']))
print("DummyClassifier Accuracy: {:.2f}".format(dummy_report['accuracy']))

# save the report as a styled table image
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')
tbl = ax.table(
    cellText=ml_report_df.round(2).values,
    colLabels=ml_report_df.columns,
    rowLabels=ml_report_df.index,
    loc='center',
    cellLoc='center',
    colColours=['#2d415b']*len(ml_report_df.columns),
    rowColours=['#2d415b']*len(ml_report_df.index)
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.3, 1.3)
# Style header
for (row, col), cell in tbl.get_celld().items():
    if row == 0 or col == -1:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2d415b')
    elif row % 2 == 0:
        cell.set_facecolor('#f2f2f2')
    else:
        cell.set_facecolor('#e6e6e6')
tbl.auto_set_column_width(col=list(range(len(ml_report_df.columns))))
plt.title('ML Polarity Detection Report', fontsize=16, pad=18, weight='bold', color='#2d415b')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results', 'ml_polarity_detection_report_tokenizer.png'), bbox_inches='tight', dpi=200)
plt.close(fig)


# save models
joblib.dump(clf_pol, os.path.join(BASE_DIR, 'results', 'model_polarity.joblib'))
joblib.dump(clf_svm, os.path.join(BASE_DIR, 'results', 'model_svm_polarity.joblib'))

# ML-based sentiment prediction
corpus = pd.read_csv(CORPUS_PATH)
# fill missing processed_text with empty string to avoid vectorizer errors
corpus['processed_text'] = corpus['processed_text'].fillna('')




# Fair timing: separate embedding and prediction
import gc

print("[DEBUG] Timing BERT embedding generation...")
start_embed = time.time()
corpus_vec = get_bert_embeddings(corpus['processed_text'], tokenizer, bert_model, device=device)
end_embed = time.time()
embedding_time = end_embed - start_embed
print(f"[DEBUG] BERT embedding generation complete in {embedding_time:.2f} seconds.")

# Logistic Regression prediction timing
start_lr = time.time()
corpus['polarity_ml'] = clf_pol.predict(corpus_vec)
end_lr = time.time()
lr_pred_time = end_lr - start_lr
print(f"[DEBUG] Logistic Regression prediction done in {lr_pred_time:.2f} seconds.")

# SVM prediction timing
start_svm = time.time()
corpus['polarity_svm'] = clf_svm.predict(corpus_vec)
end_svm = time.time()
svm_pred_time = end_svm - start_svm
print(f"[DEBUG] SVM prediction done in {svm_pred_time:.2f} seconds.")

num_records = len(corpus)
if embedding_time > 0:
    records_per_sec_embed = num_records / embedding_time
else:
    records_per_sec_embed = float('inf')

if lr_pred_time > 0:
    records_per_sec_lr = num_records / lr_pred_time
else:
    records_per_sec_lr = float('inf')

if svm_pred_time > 0:
    records_per_sec_svm = num_records / svm_pred_time
else:
    records_per_sec_svm = float('inf')

corpus['polarity_ml'] = corpus['polarity_ml'].map({1: 'POSITIVE', 0: 'NEGATIVE'})
corpus['polarity_svm'] = corpus['polarity_svm'].map({1: 'POSITIVE', 0: 'NEGATIVE'})

gc.collect()


# dummyClassifier prediction on full corpus
dummy_preds_corpus = dummy_clf.predict(corpus_vec)
dummy_preds_corpus = pd.Series(dummy_preds_corpus).map({1: 'POSITIVE', 0: 'NEGATIVE'})


print("\n--- Prediction Distribution Comparison on the rest of the dataset ---")
order = ['NEGATIVE', 'POSITIVE']
print("Logistic Regression prediction distribution:")
print(corpus['polarity_ml'].value_counts().reindex(order, fill_value=0))
print("\nSVM prediction distribution:")
print(corpus['polarity_svm'].value_counts().reindex(order, fill_value=0))
print("\nDummyClassifier (Random) prediction distribution:")
print(dummy_preds_corpus.value_counts().reindex(order, fill_value=0))

 # save output
corpus.to_csv(os.path.join(BASE_DIR, 'results', 'corpus_with_sentiment.csv'), index=False)


print(f"\nBERT embedding generation complete! {num_records} records embedded in {embedding_time:.2f} seconds.")
print(f"Records embedded per second: {records_per_sec_embed:.2f}")
print(f"\nLogistic Regression classification complete! {num_records} records classified in {lr_pred_time:.2f} seconds.")
print(f"Records classified per second (Logistic Regression): {records_per_sec_lr:.2f}")
print(f"\nSVM classification complete! {num_records} records classified in {svm_pred_time:.2f} seconds.")
print(f"Records classified per second (SVM): {records_per_sec_svm:.2f}")

