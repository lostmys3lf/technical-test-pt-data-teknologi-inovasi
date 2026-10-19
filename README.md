# Prediksi Risiko Sepsis: Ensemble XGBoost + ANN

Model untuk memprediksi risiko sepsis pasien UGD dari 7 data vital dan hasil lab, berupa gabungan (soft voting ensemble) model tree-based dan Artificial Neural Network. Model disajikan lewat REST API FastAPI dan dikemas dalam Docker.

## Struktur Repository

```
.
├── studi_kasus_nuha_care.ipynb
├── main.py
├── artifacts/
│   ├── preprocessor.joblib
│   ├── model_a_xgb.json
│   ├── model_b_ann.onnx
│   └── ensemble_config.json
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .python-version
└── Imbalanced_Classification_&_NN.ipynb
```

Dataset `sepsis_emr_data.csv` tidak disertakan di repository. Notebook membacanya dari `/content/sepsis_emr_data.csv`.

## Pendekatan

Notebook disusun mengikuti tahapan CRISP-DM. Notebook dijalankan di Google Colab (ada `!pip install`, `!zip`, dan `google.colab.files`). Untuk menjalankan ulang, unggah `sepsis_emr_data.csv` ke `/content/` lalu jalankan semua sel. Hasilnya berupa `artifacts.zip` yang isinya folder `artifacts/` di repository ini.

### 1. Data Preparation

- **Duplikat.** Dari 1753 baris, hanya 100 kombinasi nilai fitur yang unik. 1653 baris lainnya adalah salinan yang hanya berbeda di `patient_id`. Duplikat ini dihapus, karena kalau dibiarkan, pasien yang identik bisa masuk ke train fold dan test fold sekaligus. Sebelum duplikat dihapus, XGBoost mendapat skor 1.0 di semua metrik, yang berasal dari kebocoran ini, bukan dari kemampuan model. Data akhir: 100 baris (58 tidak sepsis, 42 sepsis).
- **Fitur.** Ketujuh fitur klinis dipakai semua. `patient_id` dibuang karena hanya identitas.
- **Missing value.** `lactate_level` kosong di 141 dari 1753 baris (data sebelum hapus duplikat). Diisi dengan **median**. Distribusi laktat condong ke kanan: pada data awal rata-ratanya 2.50 sedangkan mediannya 2.1, dengan nilai tertinggi 5.3. Nilai laktat tinggi berasal dari pasien sepsis dan tidak dibuang (lihat Outlier), sehingga rata-rata ikut tertarik naik. Median tidak terpengaruh nilai-nilai itu, jadi data yang kosong tidak diisi dengan nilai yang condong ke arah sepsis. Median dihitung dari training fold saja.
- **Tanpa missing indicator.** Semua pasien yang laktatnya kosong ternyata tidak sepsis (uji chi-square, p ≈ 1.5e-25). Kalau diberi kolom penanda, model akan belajar bahwa "laktat kosong berarti aman". Di UGD hal ini berbahaya, karena laktat bisa kosong hanya karena hasil lab belum keluar.
- **Outlier.** Tidak dibuang. Semua nilai masih masuk akal secara medis, dan nilai ekstrem (misalnya heart rate tinggi) justru merupakan tanda klinis sepsis.
- **Scaling.** `StandardScaler`. Diperlukan untuk ANN. XGBoost tidak terpengaruh skala, tetapi kedua model memakai input yang sama supaya preprocessing di API cukup satu jalur.
- Imputer dan scaler digabung dalam satu `sklearn.pipeline.Pipeline` dan **di-fit hanya pada training fold** di setiap fold, sehingga tidak ada data leakage.

### 2. Modeling

- **Model A (tree-based):** `XGBClassifier(objective='binary:logistic', random_state=42)`, hyperparameter default.
- **Model B (ANN, TensorFlow/Keras):** `Dense(16, relu)` → `Dropout(0.5)` → `Dense(1, sigmoid)`, optimizer Adam, loss `binary_crossentropy`, 100 epoch. Jaringan dibuat kecil karena setiap train fold hanya berisi 80 baris.
- **Ensemble:** soft voting, `p = 0.5 × p_XGBoost + 0.5 × p_ANN`.

**Alasan bobot 50/50:** AUC-ROC hasil cross-validation kedua model hampir sama (XGBoost 0.9754, ANN 0.9828). Bobot yang dihitung proporsional terhadap AUC hanya menghasilkan 0.498 vs 0.502, praktis sama dengan 50/50. Bobot 50/50 juga tidak dipilih berdasarkan skor cross-validation yang sama dengan yang dilaporkan, sehingga skor ensemble tidak menjadi terlalu optimis.

### 3. Evaluation

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Di setiap fold, preprocessor dan kedua model dilatih ulang dari awal pada training fold, lalu memprediksi test fold. Probabilitas seluruh 100 pasien dikumpulkan dari fold tempat pasien tersebut menjadi data test, lalu metrik dihitung dari kumpulan prediksi itu. Threshold klasifikasi 0.5.

## Performa Model Ensemble

| Metrik | Ensemble |
|---|---|
| AUC-ROC | 0.9828 |
| Precision | 0.9545 |
| Recall | 1.0000 |

Confusion matrix ensemble (100 pasien):

| | Prediksi 0 | Prediksi 1 |
|---|---|---|
| **Aktual 0** | 56 | 2 |
| **Aktual 1** | 0 | 42 |

Semua 42 pasien sepsis terdeteksi (recall 1.0). Kesalahannya hanya 2 pasien tidak sepsis yang diprediksi berisiko. Untuk konteks UGD, jenis kesalahan ini lebih bisa diterima daripada pasien sepsis yang terlewat.

AUC-ROC model individual: XGBoost 0.9754, ANN 0.9828.

**Catatan:** skor setinggi ini lebih banyak disebabkan oleh datanya daripada oleh modelnya. `wbc_count` sendirian sudah memisahkan kelas dengan sempurna (tidak sepsis ≤ 12.9, sepsis ≥ 13.3), begitu juga `lactate_level` (≤ 2.4 vs ≥ 2.5). Dataset kemungkinan besar sintetis, dan dengan hanya 100 baris unik, angka di atas belum tentu berlaku pada data rumah sakit sungguhan.

## Menjalankan dengan Docker

Prasyarat: Docker Desktop sudah terpasang dan berjalan.

1. Clone repository dan masuk ke foldernya.

   ```bash
   git clone <url-repo>
   cd <nama-folder-repo>
   ```

2. Build image.

   ```bash
   docker build -t sepsis-api .
   ```

3. Jalankan container.

   ```bash
   docker run -p 8000:8000 sepsis-api
   ```

4. API dapat diakses di `http://127.0.0.1:8000`. Dokumentasi interaktif (Swagger) tersedia di `http://127.0.0.1:8000/docs`.

### Menjalankan tanpa Docker (opsional)

Prasyarat: Python 3.13 dan [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run uvicorn main:app --port 8000
```

Docker memakai `requirements.txt`, sedangkan `uv` memakai `pyproject.toml` dan `uv.lock`. Versi library yang dikunci sama di keduanya.

## Contoh Request

Endpoint: `POST /predict`. `lactate_level` boleh diisi `null`, nanti diisi median dari data training oleh preprocessor.

**curl (bash / Git Bash / WSL):**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"heart_rate": 115, "respiratory_rate": 25, "temperature": 38.6, "wbc_count": 16.0, "lactate_level": 3.6, "age": 78, "num_comorbidities": 3}'
```

**curl (Windows PowerShell):**

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"heart_rate\": 115, \"respiratory_rate\": 25, \"temperature\": 38.6, \"wbc_count\": 16.0, \"lactate_level\": 3.6, \"age\": 78, \"num_comorbidities\": 3}"
```

**Python** (butuh `pip install requests`):

```python
import requests

patient = {
    "heart_rate": 115,
    "respiratory_rate": 25,
    "temperature": 38.6,
    "wbc_count": 16.0,
    "lactate_level": 3.6,
    "age": 78,
    "num_comorbidities": 3,
}
response = requests.post("http://127.0.0.1:8000/predict", json=patient)
print(response.json())
```

Respons berisi dua field:

- `sepsis_risk_prediction`: 1 jika berisiko sepsis, 0 jika tidak (probabilitas ensemble ≥ 0.5)
- `risk_probability`: probabilitas sepsis hasil ensemble (0 sampai 1)

## Pilihan Desain dan Asumsi

- **ANN disimpan dalam format ONNX** dan dijalankan dengan `onnxruntime`. API tidak membutuhkan TensorFlow sama sekali, sehingga image Docker jauh lebih kecil.
- **Alasan memilih ONNX Runtime.** ONNX adalah format model terbuka yang tidak terikat satu framework. Model dari Keras, PyTorch, dan scikit-learn bisa diekspor ke ONNX, lalu dijalankan dengan ONNX Runtime di berbagai platform (server, desktop, Android, iOS, browser) dan hardware (CPU, GPU). Dengan begitu model tidak terkunci di Keras. Pada proyek ini API hanya berjalan di CPU di dalam Docker.
- **uv sebagai package manager** untuk menjalankan proyek di lokal. uv ditulis dengan Rust dan jauh lebih cepat daripada pip dalam memasang dependency. `uv.lock` mengunci versi seluruh dependency sehingga environment bisa direproduksi. `uv run` menjalankan perintah di dalam environment proyek tanpa perlu mengaktifkan virtualenv. Dockerfile tetap memakai pip dengan `requirements.txt`.
- **XGBoost disimpan dengan `save_model` (JSON)**, format bawaan XGBoost yang lebih aman dipindah antarversi dibanding pickle.
- **Preprocessor disimpan sebagai satu file** (`Pipeline` imputer + scaler) dengan `joblib`, sehingga API cukup memanggil `transform`.
- **Konfigurasi ensemble terpisah** di `ensemble_config.json` (urutan fitur, bobot, threshold). Jika model dilatih ulang dan bobot berubah, kode API tidak perlu diubah.
- **Model final** untuk API dilatih ulang pada seluruh data (100 baris) setelah evaluasi k-fold selesai.
- **Versi library dikunci** (`numpy`, `scikit-learn`, `xgboost`, `onnxruntime`) mengikuti environment training, dan base image memakai Python 3.13 yang sama, agar artefak bisa di-load tanpa masalah kompatibilitas.
- **Tanpa hyperparameter tuning dan early stopping.** Tidak diminta soal, dan dengan data sekecil ini tuning berisiko overfit ke fold tertentu.
- **Threshold 0.5** (default). Karena recall sudah 1.0 pada threshold ini, threshold tidak diturunkan.
- **Asumsi:** duplikat dianggap artefak pembuatan data, bukan pasien berbeda, karena hampir tidak mungkin belasan pasien memiliki 7 nilai vital dan lab yang sama persis.