# Report – Voice-Enabled Fitness Chatbot (Speech Recognition + Deep Learning)

**Live link:** `<paste your Vercel URL here>`  **Source code:** `<paste your GitHub URL here>`

## 1. Objective
Build and deploy an online chatbot that accepts voice input, converts it to text, classifies the user's intent
with a deep-learning model and replies with a correct fitness answer about exercise, diet and meals.

## 2. Dataset
A custom intent dataset (`training/intents.json`) created for this project.
* 39 intents: greeting, goodbye, thanks, about-bot, weight-loss diet, muscle-gain diet, protein intake, breakfast,
  lunch/dinner, weight gain/loss day-plans, snacks, pre-/post-workout meals, hydration, beginner/chest/back/leg/arm/shoulder/abs workouts, cardio,
  HIIT, home workout, warm-up & stretching, recovery, sleep, workout frequency, belly fat, supplements, creatine,
  calories, vegetarian protein, plateau, motivation, injury/pain, yoga, and an out-of-scope (fallback) class.
* 438 training questions (about 10–12 phrasings per intent) with one reviewed answer per intent.
* Separate test set: 86 unseen questions (`training/test_queries.json`), not used in training.

## 3. Methodology
1. **Speech recognition** – browser Web Speech API converts speech to text. Interim text is shown live; up to 5
   alternative transcripts are requested and the one the model understands best is used. Language selectable
   (English India/US/UK). Typing is available as a fallback.
2. **Pre-processing** – lowercase, remove punctuation, light plural stemming, stop-word removal.
3. **Features** – bag of words of unigrams + bigrams (vocabulary of 1,225 features).
4. **Training** – NumPy implementation of a neural network: forward pass, back-propagation, Adam optimiser
   (lr 0.01), mini-batch 32, 400 epochs, dropout 0.3, cross-entropy loss.
5. **Deployment** – trained weights exported to `model/model.json`; inference runs in JavaScript in the browser,
   hosted as a static site on Vercel (HTTPS).
6. **Safe response selection** – the network only chooses a pre-written answer. A reply is given only if
   confidence ≥ 0.60, the margin over the second-best intent ≥ 0.20, and ≥ 50% of the features are known.
   Otherwise the bot asks the user to rephrase instead of guessing.

## 4. Model architecture
```
Input (1225 bag-of-words features)
  → Dense 64, ReLU → Dropout 0.3
  → Dense 32, ReLU → Dropout 0.3
  → Dense 39, Softmax  (intent probabilities)
```
Total trainable parameters: 81,831.

## 5. Results
| Metric | Value |
|---|---|
| Training accuracy | 99.8 % |
| Held-out test accuracy (86 unseen questions) | 94.2 % (81/86) |
| Wrong answers given on test set | 0 |
| Safe "please rephrase" responses | 5 (6 %) |

The 5 non-answered cases were short chit-chat phrases with unseen words (e.g. "see you tomorrow"); the bot
declined rather than answering wrongly. Off-topic questions (weather, news, general knowledge) are correctly
routed to the out-of-scope reply. Python and browser (JavaScript) inference gave identical results.

## 6. Limitations and future work
Speech accuracy depends on the browser's speech service, accent and background noise. The model knows only the
39 intents; it can be extended by adding examples and retraining. Future work: personalised plans (BMI/calorie
calculator), multi-language support, and a larger dataset or a pretrained transformer encoder.

*Disclaimer: general fitness information, not medical advice.*
