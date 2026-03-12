from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import re
import math
import os
from urllib.parse import urlparse
from collections import Counter

app = Flask(__name__)

BASE = os.path.dirname(__file__)
scaler = joblib.load(os.path.join(BASE, "models/scaler.pkl"))
models = {
    "Random Forest":        joblib.load(os.path.join(BASE, "../models/random_forest.pkl")),
    "Logistic Regression":  joblib.load(os.path.join(BASE, "../models/logistic.pkl")),
    "SVM":                  joblib.load(os.path.join(BASE, "../models/svm.pkl")),
}
MODEL_ACCURACIES = {
    "Random Forest":        93.46,
    "Logistic Regression":  82.42,
    "SVM":                  85.81,
}
BEST_MODEL = "Random Forest"

shortners_path = os.path.join(BASE, "../data/datasets/shortners.csv")
try:
    SHORTENERS = set(pd.read_csv(shortners_path).iloc[:, 0].str.strip().tolist())
except Exception:
    SHORTENERS = set()

BRANDS = ['paypal','amazon','google','facebook','apple','microsoft','netflix',
          'instagram','twitter','linkedin','ebay','bank','wellsfargo','chase',
          'citibank','dropbox','adobe','yahoo','outlook','office365','steam',
          'youtube','whatsapp','tiktok','snapchat']

def extract_features(url: str) -> dict:
    parsed   = urlparse(url)
    hostname = parsed.netloc.lower()
    if ":" in hostname:
        hostname = hostname.split(":")[0]

    def _entropy(s):
        if not s: return 0
        cnt = Counter(s)
        return -sum((v/len(s))*math.log(v/len(s), 2) for v in cnt.values())

    parts = hostname.split('.')
    root  = '.'.join(parts[-2:]) if len(parts) >= 2 else hostname
    url_lower = url.lower()
    bim = int(any(b in url_lower and b not in root for b in BRANDS))

    tokens  = re.split(r'[/.\-_?=&#]', url)
    longest = max((len(t) for t in tokens), default=0)

    alpha   = [c for c in hostname if c.isalpha()]
    vowel_r = sum(c in 'aeiou' for c in alpha) / len(alpha) if alpha else 0

    max_rep, cur = 1, 1
    for i in range(1, len(url)):
        cur = cur + 1 if url[i] == url[i-1] else 1
        max_rep = max(max_rep, cur)

    return {
        "url_length":           len(url),
        "has_at":               int("@" in url),
        "has_dash":             url.count("-"),
        "dot":                  url.count("."),
        "has_ip":               int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", hostname))),
        "shortening_service":   int(any(s in url for s in SHORTENERS)),
        "double_slash_redirect":url.count("//") - 1 if url.count("//") > 1 else 0,
        "ssl_final_state":      int(not url.startswith("https://")),
        "https_token":          int("https" in hostname),
        "redirect_symbol":      int("//" in url[8:]),
        "suspicious_words":     sum(url_lower.count(w) for w in
                                    ["login","verify","update","secure","account",
                                     "banking","support","password","confirm"]),
        "digits":               sum(c.isdigit() for c in url),
        "path_length":          len(parsed.path),
        "suspicious_tld":       int(any(hostname.endswith(t) for t in
                                        [".xyz",".top",".tk",".ml",".ga",".cf",
                                         ".gq",".club",".site",".online",".cc",".su"])),
        "entropy":              _entropy(url),
        "domain_length":        len(parsed.netloc),
        "paramaters":           len(parsed.query.split("&")) if parsed.query else 0,
        "has_punycode":         int("xn--" in url_lower),
        "special_chars":        sum(url.count(c) for c in ["?","=","_","~","%"]),
        "brand_in_url":         sum(url_lower.count(b) for b in BRANDS),
        "brand_domain_mismatch":bim,
        "subdomain_count":      max(0, len(parts) - 2),
        "path_depth":           parsed.path.count("/"),
        "digit_ratio":          sum(c.isdigit() for c in url) / len(url) if url else 0,
        "domain_digit_count":   sum(c.isdigit() for c in hostname),
        "has_port":             int(bool(re.search(r":\d+$", parsed.netloc))),
        "tld_legitimacy":       0 if any(hostname.endswith(t) for t in
                                         [".com",".org",".net",".edu",".gov",".co",
                                          ".io",".uk",".de",".fr",".jp",".au",
                                          ".ca",".in",".it",".es",".nl"]) else 1,
        "longest_token":        longest,
        "vowel_ratio_domain":   round(vowel_r, 4),
        "hex_char_count":       len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        "max_consecutive_rep":  max_rep,
        "has_fragment":         int(bool(parsed.fragment)),
        "unique_char_ratio":    len(set(url_lower)) / len(url) if url else 0,
        "domain_entropy":       _entropy(hostname),
        "has_www":              int(parsed.netloc.lower().startswith("www.")),
    }

FEATURE_NAMES = [
    "url_length","has_at","has_dash","dot","has_ip","shortening_service",
    "double_slash_redirect","ssl_final_state","https_token","redirect_symbol",
    "suspicious_words","digits","path_length","suspicious_tld","entropy",
    "domain_length","paramaters","has_punycode","special_chars",
    "brand_in_url","brand_domain_mismatch","subdomain_count","path_depth",
    "digit_ratio","domain_digit_count","has_port","tld_legitimacy",
    "longest_token","vowel_ratio_domain","hex_char_count","max_consecutive_rep",
    "has_fragment","unique_char_ratio","domain_entropy","has_www",
]

LABEL_MAP   = {0: "Legitimate", 1: "Phishing"}
LABEL_COLOR = {0: "safe",       1: "danger"}

@app.route("/")
def index():
    return render_template("index.html", model_accuracies=MODEL_ACCURACIES)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    url  = data.get("url", "").strip()
    selected_model = data.get("model", BEST_MODEL)
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        features    = extract_features(url)
        feat_vec    = np.array([[features[f] for f in FEATURE_NAMES]])
        feat_scaled = scaler.transform(feat_vec)
        model       = models.get(selected_model, models[BEST_MODEL])
        prediction  = int(model.predict(feat_scaled)[0])
        try:
            proba         = model.predict_proba(feat_scaled)[0]
            confidence    = float(max(proba)) * 100
            phishing_prob = float(proba[1]) * 100
        except Exception:
            confidence = phishing_prob = None

        flags = []
        if features["ssl_final_state"]:         flags.append({"icon":"🔓","text":"No HTTPS"})
        if features["has_ip"]:                  flags.append({"icon":"🖥","text":"IP address as hostname"})
        if features["suspicious_tld"]:          flags.append({"icon":"⚠️","text":"Suspicious TLD"})
        if features["has_at"]:                  flags.append({"icon":"👤","text":"@ symbol in URL"})
        if features["suspicious_words"] > 0:    flags.append({"icon":"🔑","text":f"{features['suspicious_words']} suspicious keyword(s)"})
        if features["shortening_service"]:      flags.append({"icon":"🔗","text":"URL shortener detected"})
        if features["has_punycode"]:            flags.append({"icon":"🌐","text":"Punycode / IDN detected"})
        if features["brand_domain_mismatch"]:   flags.append({"icon":"🎭","text":"Brand name used outside its domain"})
        if features["brand_in_url"] > 1:        flags.append({"icon":"🏷","text":f"Multiple brand names in URL ({features['brand_in_url']})"})
        if features["subdomain_count"] > 2:     flags.append({"icon":"🔀","text":f"Excessive subdomains ({features['subdomain_count']})"})
        if features["tld_legitimacy"] == 1:     flags.append({"icon":"🌍","text":"Uncommon / unrecognised TLD"})
        if features["longest_token"] > 50:      flags.append({"icon":"📏","text":f"Abnormally long token ({features['longest_token']} chars)"})
        if features["hex_char_count"] > 3:      flags.append({"icon":"🔢","text":f"URL obfuscation detected ({features['hex_char_count']} hex chars)"})
        if features["entropy"] > 4.2:           flags.append({"icon":"📊","text":f"High entropy ({features['entropy']:.2f})"})
        if features["unique_char_ratio"] < 0.3: flags.append({"icon":"🔁","text":"Low character diversity"})

        return jsonify({
            "url":           url,
            "model":         selected_model,
            "prediction":    prediction,
            "label":         LABEL_MAP[prediction],
            "status":        LABEL_COLOR[prediction],
            "confidence":    round(confidence, 1)    if confidence    is not None else None,
            "phishing_prob": round(phishing_prob, 1) if phishing_prob is not None else None,
            "features":      {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in features.items()},
            "flags":         flags,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare", methods=["POST"])
def compare():
    data = request.get_json(force=True)
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        features    = extract_features(url)
        feat_vec    = np.array([[features[f] for f in FEATURE_NAMES]])
        feat_scaled = scaler.transform(feat_vec)
        results = []
        for name, model in models.items():
            pred = int(model.predict(feat_scaled)[0])
            try:
                proba = model.predict_proba(feat_scaled)[0]
                conf  = round(float(max(proba)) * 100, 1)
                ph    = round(float(proba[1]) * 100, 1)
            except Exception:
                conf = ph = None
            results.append({
                "model":         name,
                "accuracy":      MODEL_ACCURACIES[name],
                "prediction":    pred,
                "label":         LABEL_MAP[pred],
                "status":        LABEL_COLOR[pred],
                "confidence":    conf,
                "phishing_prob": ph,
            })
        return jsonify({"url": url, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)