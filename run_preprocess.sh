#!/bin/bash
set -e
echo "Running embedding extraction..."
python3 extract_embeddings.py --data_dir data --out_dir .
echo "Done. You can now run: streamlit run app.py"
