# %%
# Fine-Tuning BERT для Классификации Отзывов IMDB
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import torch
import torch.nn as nn
from datasets import load_dataset
from IMDBDataset import IMDBDataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, RobertaForSequenceClassification, RobertaTokenizer

# %%
# Загрузка датасета IMDB
dataset = load_dataset('imdb')
# Проверяем доступность GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Определяем модель BERT для классификации последовательностей
model = 'roberta-base'

# %%
# Проверка размера набора данных. Всего (50_000 отзывов)
print(dataset)
print(f'Размер тренировочной выборки: {len(dataset["train"])}')
print(f'Размер тестовой выборки: {len(dataset["test"])}')


# %%
# Очистка текста
def cleaning_data(examples):
    # Регулярное выражение для удаления всех символов, кроме букв и цифр
    pattern = r'[^a-zA-Zа-яА-Я0-9 ]'
    examples["text"] = [re.sub(pattern, '', text.lower()) for text in examples["text"]]
    return examples


cleaning_dataset = dataset.map(cleaning_data, batched=True)

# %%
# Предобученный токинизатор
tokenizer = RobertaTokenizer.from_pretrained(model)


# Токинизируем текст
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)


tokenized_dataset = cleaning_dataset.map(tokenize_function, batched=True)
# Преобразуем данные в тензор
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# %%
# Проверим данные
print(f'После преобразования input_ids: {tokenized_dataset['train']['input_ids']}')
print(f'После преобразования attention_mask: {tokenized_dataset['train']['attention_mask']}')
print(f'После преобразования label: {tokenized_dataset['train']['label']}')

# %%
# Разделение данных на тренировочную и тестовую выборку
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

# %%
# Инициализация датасетов
train_dataset = IMDBDataset(train_dataset)
test_dataset = IMDBDataset(test_dataset)

# %%
# Инициализация предобученной модели
model = RobertaForSequenceClassification.from_pretrained(model, num_labels=2)
# Перевод модели на GPU
model.to(device)

# %%
# Оптимизатор и функция потерь
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# %%
# Подаём данные в DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, pin_memory=False)

# %%
torch.cuda.empty_cache()
# Обучения модели
epochs = 3
for epoch in range(epochs):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_prediction = 0

    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print("⚠️ NaN в градиентах!")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss.backward()

        # Нормируем градиенты
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(logits, dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_prediction += labels.size(0)
    accuracy = correct_predictions / total_prediction
    print(f'Эпоха {epoch + 1}/{epochs}, Точность: {accuracy:.2f}, Потери: {total_loss / len(train_dataloader):.2f}')

# %%
# Сохранение модели
model_save_dir = 'roberta_base_finetuned'
model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)
print(f"Модель и токенизатор успешно сохранены в директории: {model_save_dir}")

# %%
model_save_dir = 'roberta_base_finetuned'
# Загружаем модель и сохраненной директории
saved_model = RobertaForSequenceClassification.from_pretrained(model_save_dir).to(device)
saved_model.eval()
print("Модель успешно загружена.")

correct_predictions = 0
total_prediction = 0

# Оценка модели
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        _, predicted = torch.max(logits, dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_prediction += labels.size(0)
accuracy = correct_predictions / total_prediction
print(f'Точность на тестовой выборки: {accuracy:.2f}')
