/* FitBot – voice-enabled fitness chatbot.
   Speech -> text : browser Web Speech API
   Text -> intent : trained neural network (weights in model/model.json), run in the browser
   NOTE: tokenize() must stay identical to tokenize() in training/train.py */

let MODEL = null;
let VOCAB_INDEX = null;
let STOP = null;

function loadModelObject(m) {
  MODEL = m;
  VOCAB_INDEX = new Map(m.vocab.map((w, i) => [w, i]));
  STOP = new Set(m.stopwords);
}

function tokenize(text) {
  const out = [];
  for (let w of text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/)) {
    if (!w) continue;
    if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) w = w.slice(0, -1);
    if (!STOP.has(w)) out.push(w);
  }
  return out;
}

function featuresOf(text) {
  const t = tokenize(text);
  const f = t.slice();
  for (let i = 0; i < t.length - 1; i++) f.push(t[i] + "_" + t[i + 1]);
  return f;
}

function softmax(z) {
  const mx = Math.max(...z);
  const e = z.map(v => Math.exp(v - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

function dense(inputIdx, W, b) {            // inputIdx = active (value 1) input positions
  const out = b.slice();
  for (const i of inputIdx) {
    const row = W[i];
    for (let j = 0; j < out.length; j++) out[j] += row[j];
  }
  return out;
}

function denseFull(x, W, b) {
  const out = b.slice();
  for (let i = 0; i < x.length; i++) {
    if (x[i] === 0) continue;
    const row = W[i];
    for (let j = 0; j < out.length; j++) out[j] += x[i] * row[j];
  }
  return out;
}

/* Returns {tag, confidence, sure, response}. "sure" is false when the bot should NOT guess. */
function classify(text) {
  const feats = featuresOf(text);
  const known = [...new Set(feats.filter(w => VOCAB_INDEX.has(w)))];
  const knownCount = feats.filter(w => VOCAB_INDEX.has(w)).length;
  if (known.length === 0) return { tag: "unknown", confidence: 0, sure: false };

  const h1 = dense(known.map(w => VOCAB_INDEX.get(w)), MODEL.W1, MODEL.b1).map(v => Math.max(0, v));
  const h2 = denseFull(h1, MODEL.W2, MODEL.b2).map(v => Math.max(0, v));
  const p = softmax(denseFull(h2, MODEL.W3, MODEL.b3));

  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  const sorted = p.slice().sort((a, b) => a - b);
  const conf = p[best], second = sorted[sorted.length - 2];

  const sure = conf >= MODEL.threshold &&
               (conf - second) >= MODEL.margin &&
               (knownCount / Math.max(1, feats.length)) >= MODEL.minKnown;
  const tag = MODEL.labels[best];
  return { tag, confidence: conf, sure, response: MODEL.responses[tag] };
}

/* ---- Personalising plan answers: use the kg and time period the user actually said ---- */
const NUM_WORDS = { a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8,
  nine: 9, ten: 10, eleven: 11, twelve: 12, fifteen: 15, twenty: 20, thirty: 30, forty: 40, fifty: 50 };
const NUM_RE = "(\\d+(?:\\.\\d+)?|" + Object.keys(NUM_WORDS).join("|") + ")";
const toNum = s => /^\d/.test(s) ? parseFloat(s) : NUM_WORDS[s];
const fmt = n => String(Math.round(n * 10) / 10);

function parseGoal(text) {
  const t = text.toLowerCase();
  const kg = t.match(new RegExp("\\b" + NUM_RE + "\\s*(?:kgs?|kilos?|kilograms?)\\b"));
  const per = t.match(new RegExp("\\b" + NUM_RE + "[\\s-]*(day|week|month)s?\\b"));
  if (!kg || !per) return null;
  const amount = toNum(kg[1]), n = toNum(per[1]);
  if (!(amount > 0) || !(n > 0)) return null;
  const days = n * { day: 1, week: 7, month: 30 }[per[2]];
  return { kg: amount, days, period: `${fmt(n)} ${per[2]}${n === 1 ? "" : "s"}` };
}

function planOpening(tag, text) {
  const g = parseGoal(text);
  if (!g) return null;
  const weeks = g.days / 7, rate = g.kg / weeks;
  const gain = tag === "weight_gain_plan";
  const [lo, hi] = gain ? [0.25, 0.5] : [0.5, 1];               // safe kg per week
  const within = `${fmt(lo * weeks)}-${fmt(hi * weeks)} kg`;      // realistic change in that time
  const needs = `${Math.round(g.kg / hi)}-${Math.round(g.kg / lo)} weeks`;
  const verb = gain ? "Gaining" : "Losing";
  const pace = `${lo}-${hi} kg per week`;
  const plan = gain ? "A good plan: eat 300-500 kcal above maintenance every day."
                    : "A good plan: eat 300-500 kcal below maintenance.";
  if (rate <= hi)
    return `${verb} ${fmt(g.kg)} kg in ${g.period} works out to about ${fmt(rate)} kg per week, which is within a safe, realistic pace (${pace}). ${plan}`;
  return `${verb} ${fmt(g.kg)} kg in ${g.period} (about ${fmt(rate)} kg per week) is faster than a safe pace of ${pace}. ` +
    (gain ? "Anything that fast is mostly water, food weight or fat rather than muscle. "
          : "Quick drops like that are mostly water, and crash diets are not recommended. ") +
    `At a safe pace, ${fmt(g.kg)} kg takes about ${needs}; in ${g.period} a realistic change is about ${within}. ${plan}`;
}

function parseBodyKg(text) {
  const m = text.toLowerCase().match(new RegExp("\\b" + NUM_RE + "\\s*(?:kgs?|kilos?|kilograms?)\\b"));
  const n = m ? toNum(m[1]) : NaN;
  return n >= 30 && n <= 250 ? n : null;                      // plausible body weight only
}

function personalize(tag, response, query) {
  const w = parseBodyKg(query);
  if (w && tag === "protein_intake")
    return response.replace(/For a 70 kg person that is roughly[^.]*\./,
      `For ${fmt(w)} kg that is roughly ${Math.round(w * 1.4 / 5) * 5}-${Math.round(w * 2.0 / 5) * 5} g per day (up to about ${Math.round(w * 2.2 / 5) * 5} g when building muscle).`);
  if (w && tag === "calories")
    return `At ${fmt(w)} kg, maintenance is roughly ${Math.round(w * 30)}-${Math.round(w * 33)} kcal per day for moderately active people.\n` + response;
  if (tag !== "weight_gain_plan" && tag !== "weight_loss_plan") return response;
  // older model.json files hard-code "2 kg in 10 days"; neutralise it so no model change is needed
  response = response.replace(", so 2 kg in 10 days would be mostly water and food weight rather than muscle or fat.", ".");
  const opening = planOpening(tag, query);
  if (!opening) return response;
  const lines = response.split("\n");
  lines[0] = opening;
  return lines.join("\n");
}

/* Final decision used by the UI */
function answer(text) {
  const r = classify(text);
  if (!r.sure) {
    return { ...r, tag: "not_sure", text:
      "I'm not sure I understood that, and I don't want to give you a wrong answer.\n" +
      "Please try again, for example: \"chest workout\", \"what to eat before gym\" or \"how much protein do I need\"." };
  }
  return { ...r, text: personalize(r.tag, r.response, text) };
}

if (typeof module !== "undefined") module.exports = { loadModelObject, classify, answer, parseGoal };

/* ------------------------------------------------------------------ UI */
if (typeof window !== "undefined") {
  const $ = id => document.getElementById(id);
  const micBtn = $("micBtn"), statusEl = $("status"), heardEl = $("heard"), replyEl = $("reply"),
        metaEl = $("meta"), historyEl = $("history"), langEl = $("lang"), speakEl = $("speak");

  const SUGGESTIONS = ["Chest workout", "What to eat before gym", "How much protein do I need",
    "Home workout without equipment", "How to lose belly fat", "Healthy breakfast ideas"];

  let ready = false;
  fetch("model/model.json")
    .then(r => r.json())
    .then(m => { loadModelObject(m); ready = true; statusEl.textContent = "Ready. Tap the mic and speak."; })
    .catch(() => { statusEl.textContent = "Could not load the model. Please refresh the page."; });

  SUGGESTIONS.forEach(s => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = s;
    b.onclick = () => handleQuery(s);
    $("chips").appendChild(b);
  });

  function speakOut(text) {
    if (!speakEl.checked || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.replace(/\n/g, " "));
    u.lang = langEl.value;
    window.speechSynthesis.speak(u);
  }

  function handleQuery(text) {
    text = text.trim();
    if (!text || !ready) return;
    const r = answer(text);
    heardEl.textContent = text;
    replyEl.textContent = r.text;
    heardEl.classList.remove("empty"); replyEl.classList.remove("empty");
    const unsure = r.tag === "not_sure";
    $("confBar").style.width = unsure ? "0%" : (r.confidence * 100).toFixed(0) + "%";
    replyEl.closest(".answer").classList.toggle("unsure", unsure);
    replyEl.classList.remove("pop"); void replyEl.offsetWidth; replyEl.classList.add("pop");
    metaEl.textContent = r.tag === "not_sure"
      ? "Low confidence – no answer guessed."
      : `Detected intent: ${r.tag.replace(/_/g, " ")}  |  confidence: ${(r.confidence * 100).toFixed(0)}%`;
    const y = document.createElement("div"); y.className = "you"; y.textContent = "You: " + text;
    const b = document.createElement("div"); b.className = "bot"; b.textContent = "FitBot: " + r.text;
    historyEl.prepend(b); historyEl.prepend(y);
    speakOut(r.text);
  }

  $("textForm").addEventListener("submit", e => {
    e.preventDefault();
    handleQuery($("textInput").value);
    $("textInput").value = "";
  });

  /* ---------------- Speech recognition ---------------- */
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null, listening = false;

  if (!SR) {
    micBtn.disabled = true;
    statusEl.textContent = "Voice input needs Chrome, Edge or Safari. You can still type your question below.";
  } else {
    rec = new SR();
    rec.continuous = false;
    rec.interimResults = true;
    rec.maxAlternatives = 5;

    rec.onstart = () => {
      listening = true; micBtn.classList.add("listening");
      statusEl.textContent = "Listening… speak now.";
      window.speechSynthesis && window.speechSynthesis.cancel();
    };
    rec.onend = () => {
      listening = false; micBtn.classList.remove("listening");
      if (statusEl.textContent.startsWith("Listening")) statusEl.textContent = "Ready. Tap the mic and speak.";
    };
    rec.onerror = e => {
      const msg = {
        "not-allowed": "Microphone permission denied. Allow microphone access in your browser and try again.",
        "service-not-allowed": "Microphone permission denied. Allow microphone access and try again.",
        "no-speech": "I didn't hear anything. Tap the mic and try again.",
        "audio-capture": "No microphone found.",
        "network": "Speech service needs an internet connection."
      }[e.error] || "Speech error: " + e.error;
      statusEl.textContent = msg;
    };
    rec.onresult = e => {
      const res = e.results[e.results.length - 1];
      if (!res.isFinal) {                       // live preview while speaking
        heardEl.textContent = res[0].transcript + " …";
        heardEl.classList.remove("empty");
        return;
      }
      // Several possible transcripts: use the one the neural network understands best.
      let bestText = res[0].transcript, bestScore = -1;
      for (let i = 0; i < res.length; i++) {
        const t = res[i].transcript;
        const c = classify(t);
        const score = c.sure ? c.confidence : -1;
        if (score > bestScore) { bestScore = score; bestText = t; }
      }
      statusEl.textContent = "Ready. Tap the mic and speak.";
      handleQuery(bestText);
    };

    micBtn.addEventListener("click", () => {
      if (!ready) return;
      if (listening) { rec.stop(); return; }
      rec.lang = langEl.value;
      try { rec.start(); } catch (_) { /* already started */ }
    });
  }
}
