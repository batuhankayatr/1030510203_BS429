import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
 
csvPath = r"otu.csv"
dataset = pd.read_csv(csvPath, dtype=str)

X = dataset.iloc[1:, :].T
y = dataset.iloc[:1, :].T.squeeze()


le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2500, random_state=2)

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


Matris = confusion_matrix(y_test, y_pred)
classificationReport = classification_report(y_test, y_pred)
print("Matrisler:\n", Matris)
print("Sınıflandırma Raporu:\n", classificationReport)


accuracy = accuracy_score(y_test, y_pred)
sensitivity = Matris[0, 0] / (Matris[0, 0] + Matris[0, 1])
print("Tutarlılık oranı:", accuracy, "\n")
print('Hasssaslık : ', sensitivity)


specificity = Matris[1, 1] / (Matris[1, 0] + Matris[1, 1])
ROC_AUC = roc_auc_score(y_test, y_pred)
print('Özgüllük : ', specificity)
print('ROC AUC : {:.4f}'.format(ROC_AUC))