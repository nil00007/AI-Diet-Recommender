# AI‑Based Personalized Diet Recommender System
Reproducible codebase for SEN4018 Spring 24‑25 project.

## Kurulum
git clone <repo‑url>
cd diet_recommender_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py            # model eğitimi
python src/evaluate.py         # test metrikleri
python app.py                  # Gradio arayüzü
