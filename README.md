# FitBot – Voice-Enabled Fitness Chatbot

Speak (or type) a fitness question → speech is converted to text → a neural network detects the intent →
FitBot shows the recognized speech and a curated, safe answer (and reads it aloud).

* **Speech recognition:** browser Web Speech API (Chrome / Edge / Safari)
* **Deep learning:** feed-forward neural network (Dense 64 → 32 → 39 softmax) trained on 438 labelled questions
* **Runs fully in the browser** – static site, no backend, no API keys, free on Vercel

## Folder structure
```
index.html, style.css, app.js     the web app
model/model.json                  trained network (weights + vocabulary + answers)
training/intents.json             dataset (intents, example questions, answers)
training/test_queries.json        unseen test questions
training/train.py                 training + evaluation script
REPORT.md                         brief report for submission
```

## Deploy (GitHub → Vercel)
1. Create a new GitHub repository and push this folder:
   ```
   git init
   git add .
   git commit -m "FitBot voice fitness chatbot"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. On vercel.com → **Add New → Project** → import the repo.
3. Framework Preset: **Other**. Leave Build Command and Output Directory empty. Click **Deploy**.
4. Open the `https://<project>.vercel.app` link and allow microphone access. Submit this link.

Vercel serves over HTTPS, which the microphone requires.

## Run locally
Do not open index.html by double-clicking (the model file must be fetched). Use:
```
python3 -m http.server 8000
```
then open http://localhost:8000 in Chrome or Edge.

## Retrain (optional)
```
pip install numpy
cd training
python train.py
```
Edit `training/intents.json` to add questions/answers, then retrain; `model/model.json` is regenerated.

## Why it doesn't give wrong answers
Answers are written in advance for each intent (the model only *selects* one – it never generates text).
The bot answers only if the network is confident (≥ 60%), clearly ahead of the second-best intent, and most
words are known. Otherwise it asks the user to rephrase. Off-topic questions get a "fitness topics only" reply.

For weight-gain / weight-loss plan questions the opening line is built from the kg and time period the user actually said
(e.g. "5 kg in 5 days"), compared with a safe pace of 0.25-0.5 kg/week (gain) or 0.5-1 kg/week (loss).
