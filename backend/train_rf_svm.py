from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import numpy as np
import os
from backend.signal_utils import load_signal

# Формируем обучающую выборку из файлов в backend/signal_library/
X_train = []
y_train = []
class_to_idx = {}

signal_lib_dir = "backend/signal_library"
for idx, class_name in enumerate(sorted(os.listdir(signal_lib_dir))):
    class_dir = os.path.join(signal_lib_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    class_to_idx[class_name] = idx
    for fname in os.listdir(class_dir):
        if fname.endswith((".npy", ".wav", ".bin", ".raw")):
            fpath = os.path.join(class_dir, fname)
            data = load_signal(fpath)
            X_train.append(np.abs(data[:1024]))
            y_train.append(idx)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Обучающих примеров: {len(X_train)}")
print(f"Классы: {class_to_idx}")

# Обучаем Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
joblib.dump(rf, "backend/rf_model.pkl")

# Обучаем SVM
svm = SVC(probability=True)
svm.fit(X_train, y_train)
joblib.dump(svm, "backend/svm_model.pkl")

print("Модели RF и SVM сохранены в backend/")

def create_class_folder(class_name, base_dir="backend/signal_library"):
    """
    Создаёт папку для нового класса сигнала, если её ещё нет.
    """
    class_dir = os.path.join(base_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    print(f"Папка для класса '{class_name}' создана по пути: {class_dir}")

def get_class_stats(base_dir="backend/signal_library"):
    """
    Возвращает словарь: {класс: количество файлов}
    """
    stats = {}
    for class_name in sorted(os.listdir(base_dir)):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        count = len([f for f in os.listdir(class_dir) if f.endswith((".npy", ".wav", ".bin", ".raw"))])
        stats[class_name] = count
    return stats

if __name__ == "__main__":
    # Пример: создать новую папку-класс
    create_class_folder("NEW_CLASS")

    # Пример: вывести статистику по классам
    stats = get_class_stats()
    print("Статистика по классам сигналов:")
    for cls, count in stats.items():
        print(f"  {cls}: {count} файлов")