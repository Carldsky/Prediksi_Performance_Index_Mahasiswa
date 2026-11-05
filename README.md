# Milestone 1 (UTS) – Prediksi Performance Index Mahasiswa

Proyek regresi dengan classical machine learning untuk memprediksi **Performance Index** (skala 10–100) dari kebiasaan belajar mahasiswa.

## Dataset

- Nama: Student Performance (Multiple Linear Regression)
- Sumber: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression
- File: `Student_Performance.csv` (10.000 baris, 6 kolom)
- Jika file tidak disertakan, unduh dari tautan di atas lalu letakkan di folder yang sama dengan `main.py`.

## Struktur folder

```
Prediksi_Performance_Index_Mahasiswa/
├── main.ipynb                    # seluruh eksperimen (bagian 0–13)
├── Student_Performance.csv    # dataset
├── requirements.txt
├── README.md
└── outputs/                   # dibuat otomatis saat main.py dijalankan
    ├── figures/               # 9 grafik EDA dan evaluasi
    ├── tabel_hasil_eksperimen.csv
    ├── log_eksperimen.csv
    └── ... (CSV ringkasan lainnya)
```

## Cara menjalankan

```bash
pip install -r requirements.txt
python main.py
```

Waktu eksekusi sekitar 30 detik. Seluruh hasil memakai `random_state = 42`, sehingga angka akan sama setiap kali dijalankan dengan versi library yang sama (lihat `outputs/versi_lingkungan.txt`).

## Pemetaan laporan ke kode dan artefak

| Bagian laporan | Bagian di `main.py` | Artefak di `outputs/` |
|---|---|---|
| B. Dataset & data understanding | 1, 2 | `ringkasan_kualitas_data.csv`, `cek_rentang_nilai.csv` |
| C. EDA | 4 | `eda_*.csv`, `figures/01`–`06` |
| D. Preprocessing & pencegahan leakage | 2, 3, 5 | `info_pembagian_data.csv` |
| E. Feature engineering | 6 | baris E09–E11 di `tabel_hasil_eksperimen.csv` |
| F. Baseline & model kandidat | 7 | `tabel_hasil_eksperimen.csv` |
| G. Desain evaluasi | 3, 8, 9 | `hasil_validasi_silang.csv` |
| H. Hasil & analisis | 9, 10, 11 | `tabel_hasil_eksperimen.csv`, `ringkasan_residual.csv`, `koefisien_linear_regression.csv`, `permutation_importance.csv`, `figures/07`–`09` |
| I. Catatan eksperimen | 12 | `log_eksperimen.csv` |

## Daftar eksperimen

| ID | Model | Jenis |
|---|---|---|
| E01 | Dummy Regressor (mean) | Baseline |
| E02 | Linear Regression, hanya Previous Scores | Baseline |
| E03 | Linear Regression | Kandidat |
| E04 | Ridge Regression | Kandidat |
| E05 | KNN Regressor | Kandidat |
| E06 | Decision Tree | Kandidat |
| E07 | Random Forest | Kandidat |
| E08 | Gradient Boosting | Kandidat |
| E09 | Linear Regression + fitur interaksi & rasio | Feature engineering |
| E10 | Linear Regression + polinomial derajat 2 | Feature engineering |
| E11 | Random Forest + fitur interaksi & rasio | Feature engineering |
