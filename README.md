# FitBot – Voice-Enabled Fitness Chatbot

**Live demo:** https://fitbot-psi.vercel.app/

FitBot is an online voice assistant for fitness questions. Speak (or type) a question about workouts, meals or diet. Your speech is converted to text, a neural network detects what you are asking, and FitBot shows both the recognized speech and its answer, and can read the answer aloud.

## Features
- Voice input with live transcription, in English (India, US or UK)
- Deep-learning intent classification across 39 fitness topics
- Recognized speech and chatbot response shown side by side, with a confidence bar
- Personalized answers: weight gain/loss plans use the kg and time period you say, and protein and calorie answers use your body weight
- Answers read aloud (optional), suggested questions and conversation history
- Responsive green-and-black interface that fits the full screen

## How it works
1. **Speech recognition:** the browser Web Speech API converts speech to text.
2. **Intent classification:** the text is turned into unigram and bigram features and passed through a feed-forward neural network (Dense 64 → 32 → 39, softmax) trained on 438 labelled questions.
3. **Response:** FitBot returns the reviewed answer for the detected intent. If the network is not confident enough, it asks the user to rephrase instead of guessing.

## Results
- Training accuracy: 99.8%
- Held-out test accuracy: 94.2% (81/86 unseen questions), with 0 wrong answers

## Project structure
```
index.html, style.css, app.js     web app
model/model.json                  trained network (weights, vocabulary, answers)
training/intents.json             dataset: intents, example questions, answers
training/test_queries.json        unseen test questions
training/train.py                 training and evaluation script
REPORT.md                         project report
```

*General fitness information only, not medical advice.*
