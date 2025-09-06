
# ECG Viewer & Analyzer (Streamlit App)

## 📂 Estructura

```
ecg-app/
├─ app.py                 # App principal Streamlit
├─ ecg_utils.py           # Funciones auxiliares
├─ requirements.txt       # Dependencias
├─ README.md              # Instrucciones
├─ .streamlit/config.toml # Tema de UI
├─ data/                  # Aquí van los .npz grandes (ignorado en git)
│  └─ index.csv           # Índice con metadatos y rutas
├─ samples/               # 1–3 .npz de ejemplo (subidos al repo para Cloud)
├─ scripts/               # Utilidades para procesar dataset
│  ├─ build_index.py
│  └─ validate_npz.py
└─ .gitignore
```

## 🚀 Uso local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Serialización e indexado
Recuerda que si tienes un dataset que tiene .hea o .mat, serializarlo a .npz
e indexarlos a .csv para volver más ligero el repositorio.

```bash
Serialización:
python prepare_slices_patch.py --raw_dir ["dirección archivo de entrada"] --out_dir ["dirección archivo de salida"] --lead II --win_s 8 --fs_out 500 --max_segments 0
Ejemplo: python prepare_slices_patch.py --raw_dir "data\raw\WFDBRecords\02" --out_dir "data\slices\g02" --lead II --win_s 8 --fs_out 500 --max_segments 0
Indexado:
python build_index_from_npz.py --src SRC [--out OUT] [--seg-window SEG_WINDOW] [--seg-step SEG_STEP]
Ejemplo: python build_index_from_npz.py --src data/ --out data/index.csv
```

- Coloca tus `.npz` en `data/`.
- Genera/actualiza el índice con:
```bash
python scripts/build_index_from_npz.py --src data/ --out data/index.csv
```

## ☁️ Streamlit Cloud
- No subas `data/` pesada.  
- Sube solo `samples/` con 1–3 archivos chicos `.npz` + `samples/index.csv`.  
- La app usará `data/index.csv` si existe, o `samples/index.csv` como fallback.

