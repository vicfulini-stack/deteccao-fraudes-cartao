import pandas as pd

url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'

df = pd.read_csv(url)

print(df.head())

print (df["Class"].value_counts())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Preparação (Scaling no Amount)
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

# 2. Divisão
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Modelo Supervisionado com peso balanceado
# O class_weight='balanced' ajusta automaticamente o peso das classes para o modelo dar mais importância à fraude
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 4. Predição e Avaliação
y_pred = model.predict(X_test)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.show()