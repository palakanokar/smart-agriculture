"""
explainer.py
------------
Creates a human-readable explanation for a prediction.
- If the environment variable GEMINI_API_KEY is set (free tier available),
  it asks Google Gemini for a short explanation.
- Otherwise (default) it uses a free RULE-BASED explanation that compares
  the user's inputs with the average conditions of the predicted crop.
"""
import os
import json
import urllib.request

LABELS = {
    "N": ("Nitrogen", "kg/ha"), "P": ("Phosphorus", "kg/ha"), "K": ("Potassium", "kg/ha"),
    "temperature": ("Temperature", "°C"), "humidity": ("Humidity", "%"),
    "ph": ("Soil pH", ""), "rainfall": ("Rainfall", "mm"),
}


def rule_based_explanation(crop, inputs, crop_means):
    """Compare each input with the typical value for this crop in the dataset."""
    ideal = crop_means.get(crop, {})
    points = []
    for key, (name, unit) in LABELS.items():
        user_val, avg = inputs[key], ideal.get(key)
        if avg is None:
            continue
        diff = (user_val - avg) / avg if avg else 0
        if abs(diff) <= 0.25:
            verdict = "matches well with"
        elif diff > 0:
            verdict = "is higher than"
        else:
            verdict = "is lower than"
        points.append(f"{name} ({user_val:g}{unit}) {verdict} the typical {avg:g}{unit} for {crop}.")

    ph = inputs["ph"]
    soil = "acidic" if ph < 6 else "alkaline" if ph > 7.5 else "neutral"
    summary = (
        f"The model recommends {crop.title()} because your soil nutrients and climate "
        f"most closely resemble the conditions where {crop} grows well in the dataset. "
        f"Your soil is {soil} (pH {ph:g})."
    )
    return {"source": "Rule-based", "summary": summary, "points": points}


def gemini_explanation(crop, inputs, api_key):
    """Optional: ask the free Gemini API for a 3-sentence explanation."""
    prompt = (f"In 3 simple sentences, explain to a farmer why {crop} suits these conditions: "
              f"{json.dumps(inputs)} (N,P,K in kg/ha, temperature °C, humidity %, rainfall mm).")
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-1.5-flash:generateContent?key={api_key}")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.load(r)
    return data["candidates"][0]["content"]["parts"][0]["text"]


def explain(crop, inputs, crop_means):
    result = rule_based_explanation(crop, inputs, crop_means)
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            result["summary"] = gemini_explanation(crop, inputs, api_key)
            result["source"] = "Generative AI (Gemini)"
        except Exception:
            pass  # silently fall back to rule-based text
    return result
