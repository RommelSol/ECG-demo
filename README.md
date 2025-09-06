
# ECG Viewer & Analyzer (Streamlit App)

Este repositorio implementa la app final para visualizar y analizar ECGs (O1 y O2).

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

- Coloca tus `.npz` en `data/`.
- Genera/actualiza el índice con:
```bash
python scripts/build_index.py --src data/ --out data/index.csv
```

## ☁️ Streamlit Cloud
- No subas `data/` pesada.  
- Sube solo `samples/` con 1–3 archivos chicos `.npz` + `samples/index.csv`.  
- La app usará `data/index.csv` si existe, o `samples/index.csv` como fallback.

