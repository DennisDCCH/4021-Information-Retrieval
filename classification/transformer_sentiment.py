import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import time
import matplotlib.pyplot as plt
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_PATH = os.path.join(BASE_DIR, 'evaluation', 'evaluation_dataset.xlsx')

eval_df = pd.read_excel(EVAL_PATH)
eval_df['text'] = (eval_df['summary'].fillna('') + ' ' + eval_df['content'].fillna('')).str.strip()
eval_df['ground_truth_polarity'] = eval_df['ground_truth_polarity'].astype(str).str.strip().str.upper()
eval_df['polarity'] = eval_df['ground_truth_polarity'].map({'POSITIVE': 1, 'NEGATIVE': 0, 'NEUTRAL': 2})
df_pol = eval_df[eval_df['polarity'] != 2]

# split
test_size = 0.2
X = df_pol['text'].tolist()
y = df_pol['polarity'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=99)

# transformer setup
MODEL_NAME = 'distilbert-base-uncased' 
num_labels = 2

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples, truncation=True, padding='max_length', max_length=128)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=128)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = SimpleDataset(X_train, y_train)
test_dataset = SimpleDataset(X_test, y_test)


# load from latest checkpoint if available, else from base model
checkpoint_dir = os.path.join(BASE_DIR, 'transformer_results')
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*')), key=lambda x: int(x.split('-')[-1]))
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    print(f"[INFO] Loading model from latest checkpoint: {latest_checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint)
else:
    print(f"[INFO] No checkpoint found. Loading base model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, 'transformer_model'),
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir=os.path.join(BASE_DIR, 'logs'),
    logging_steps=10,
    disable_tqdm=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()


# evaluation
preds = trainer.predict(test_dataset)
y_pred = preds.predictions.argmax(axis=1)
report = classification_report(y_test, y_pred, target_names=['NEGATIVE', 'POSITIVE'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nTransformer Model Polarity Detection Report:")
print(report_df.to_string(float_format="{:.2f}".format))

# save styled table image of report
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')
tbl = ax.table(
    cellText=report_df.round(2).values,
    colLabels=report_df.columns,
    rowLabels=report_df.index,
    loc='center',
    cellLoc='center',
    colColours=['#1f77b4']*len(report_df.columns),
    rowColours=['#1f77b4']*len(report_df.index)
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.3, 1.3)
for (row, col), cell in tbl.get_celld().items():
    if row == 0 or col == -1:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#1f77b4')
    elif row % 2 == 0:
        cell.set_facecolor('#e6f2fa')
    else:
        cell.set_facecolor('#c6e2fa')
tbl.auto_set_column_width(col=list(range(len(report_df.columns))))
plt.title('Transformer Polarity Detection Report', fontsize=16, pad=18, weight='bold', color='#1f77b4')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'results', 'transformer_polarity_detection_report.png'), bbox_inches='tight', dpi=200)
plt.close(fig)

# save predictions and perform random accuracy test on full corpus
CORPUS_PATH = os.path.join(BASE_DIR, '..', 'preprocessing', 'Temp_preprocessed_data.csv')
corpus = pd.read_csv(CORPUS_PATH)
corpus['processed_text'] = corpus['processed_text'].fillna('')

# corpus labelling
num_records = len(corpus)
t0 = time.time()
inputs = tokenizer(list(corpus['processed_text']), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
t1 = time.time()
with torch.no_grad():
    outputs = model(**inputs)
    preds_corpus = outputs.logits.argmax(dim=1).cpu().numpy()
t2 = time.time()
tokenize_time = t1 - t0
classify_time = t2 - t1
corpus['polarity_transformer'] = pd.Series(preds_corpus).map({1: 'POSITIVE', 0: 'NEGATIVE'})

records_per_sec_tokenize = num_records / tokenize_time if tokenize_time > 0 else float('inf')
records_per_sec_classify = num_records / classify_time if classify_time > 0 else float('inf')

order = ['NEGATIVE', 'POSITIVE']
dist_output = []
dist_output.append("\n--- Transformer Prediction Distribution on Full Corpus ---")
dist_output.append(str(corpus['polarity_transformer'].value_counts().reindex(order, fill_value=0)))
dist_output.append(f"\nTokenization time: {tokenize_time:.2f} seconds ({records_per_sec_tokenize:.2f} records/sec)")
dist_output.append(f"Classification time: {classify_time:.2f} seconds ({records_per_sec_classify:.2f} records/sec)")
dist_output.append(f"\nTransformer classified {num_records} records in {tokenize_time + classify_time:.2f} seconds")

for line in dist_output:
    print(line)

# Save output
corpus.to_csv(os.path.join(BASE_DIR, 'results', 'corpus_with_sentiment_transformer.csv'), index=False)
with open(os.path.join(BASE_DIR, 'results', 'transformer_distribution_output.txt'), 'w', encoding='utf-8') as f:
    for line in dist_output:
        f.write(line + '\n')

