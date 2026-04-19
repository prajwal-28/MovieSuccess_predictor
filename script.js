/**
 * Movie Success Predictor — Frontend Logic
 * =========================================
 * Handles:
 *  - Auto-calculation of ROI from budget + revenue
 *  - Form validation with inline feedback
 *  - POST /predict API call (JSON)
 *  - Displaying prediction result and probability bars
 *  - Error handling for network and API failures
 */

"use strict";

const API_URL = "http://127.0.0.1:5000/predict";

// ── DOM References ─────────────────────────────────────────────
const form        = document.getElementById("predict-form");
const submitBtn   = document.getElementById("predict-btn");
const errorBanner = document.getElementById("error-banner");
const errorText   = document.getElementById("error-text");
const resultCard  = document.getElementById("result-card");
const resultIcon  = document.getElementById("result-icon");
const resultVerdict = document.getElementById("result-verdict");
const probSection = document.getElementById("prob-section");
const roiHint     = document.getElementById("roi-hint");
const roiCalcVal  = document.getElementById("roi-calc-val");
const roiInput    = document.getElementById("roi");
const budgetInput = document.getElementById("budget");
const revenueInput= document.getElementById("revenue");

// ── Config Maps ─────────────────────────────────────────────────
const RESULT_CONFIG = {
  Hit:     { cls: "hit",  icon: "🏆", barClass: "hit-bar" },
  Average: { cls: "avg",  icon: "🎭", barClass: "avg-bar" },
  Flop:    { cls: "flop", icon: "💸", barClass: "flop-bar" },
};

const REQUIRED_FIELDS = [
  "budget", "revenue", "rating", "votes",
  "runtime", "year", "roi", "genre",
];

// ── ROI Auto-Calculate ──────────────────────────────────────────
/**
 * Watches budget + revenue inputs and auto-fills ROI,
 * showing a tooltip with the computed value.
 */
function updateRoiHint() {
  const budget  = parseFloat(budgetInput.value);
  const revenue = parseFloat(revenueInput.value);

  if (budget > 0 && revenue >= 0) {
    const roi = (revenue / budget).toFixed(4);
    roiInput.value   = roi;
    roiCalcVal.textContent = roi;
    roiHint.style.display  = "block";
  } else {
    roiHint.style.display = "none";
  }
}

budgetInput.addEventListener("input", updateRoiHint);
revenueInput.addEventListener("input", updateRoiHint);

// ── Validation ──────────────────────────────────────────────────
/**
 * Validate all required fields.
 * Adds / removes the `.invalid` CSS class for visual feedback.
 *
 * @returns {{ valid: boolean, errors: string[] }}
 */
function validateForm() {
  const errors = [];

  REQUIRED_FIELDS.forEach((fieldId) => {
    const el = document.getElementById(fieldId);
    const val = el ? el.value.trim() : "";

    if (!val) {
      el && el.classList.add("invalid");
      errors.push(`'${fieldId}' is required.`);
      return;
    }
    el && el.classList.remove("invalid");

    // Range checks
    if (fieldId === "rating") {
      const n = parseFloat(val);
      if (n < 0 || n > 10) {
        el.classList.add("invalid");
        errors.push("Rating must be between 0 and 10.");
      }
    }

    if (["budget", "revenue", "votes", "runtime"].includes(fieldId)) {
      if (parseFloat(val) < 0) {
        el.classList.add("invalid");
        errors.push(`'${fieldId}' cannot be negative.`);
      }
    }

    if (fieldId === "year") {
      const y = parseInt(val, 10);
      if (y < 1900 || y > 2100) {
        el.classList.add("invalid");
        errors.push("Year must be between 1900 and 2100.");
      }
    }
  });

  return { valid: errors.length === 0, errors };
}

// ── UI State Helpers ────────────────────────────────────────────
function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.classList.toggle("loading", isLoading);
}

function showError(message) {
  errorText.textContent = message;
  errorBanner.classList.add("visible");
  // Auto-hide after 10 seconds
  clearTimeout(showError._timer);
  showError._timer = setTimeout(() => {
    errorBanner.classList.remove("visible");
  }, 10000);
}

function hideError() {
  errorBanner.classList.remove("visible");
}

function hideResult() {
  resultCard.classList.remove("visible", "hit", "avg", "flop");
  resultCard.style.display = "none";
}

/**
 * Render the prediction result card.
 *
 * @param {string} prediction - "Hit" | "Average" | "Flop"
 * @param {Object|null} probabilities - e.g. { Hit: 1.0, Average: 0.0, Flop: 0.0 }
 */
function showResult(prediction, probabilities) {
  const cfg = RESULT_CONFIG[prediction] || RESULT_CONFIG["Average"];

  // Strip old state classes
  resultCard.className = "result-card";
  resultCard.classList.add(cfg.cls, "visible");
  resultCard.style.display = "block";

  resultIcon.textContent    = cfg.icon;
  resultVerdict.textContent = prediction;

  // Probability bars
  probSection.innerHTML = "";

  if (probabilities && Object.keys(probabilities).length > 0) {
    const labelEl = document.createElement("p");
    labelEl.className = "prob-label";
    labelEl.textContent = "Confidence breakdown";
    probSection.appendChild(labelEl);

    const order = ["Hit", "Average", "Flop"];
    order.forEach((label) => {
      const pct = probabilities[label] !== undefined
        ? Math.round(probabilities[label] * 100)
        : 0;

      const barCls = RESULT_CONFIG[label]?.barClass ?? "";

      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-name">${label}</span>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill ${barCls}" data-pct="${pct}"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
      `;
      probSection.appendChild(row);
    });

    // Animate bars on next frame
    requestAnimationFrame(() => {
      probSection.querySelectorAll(".prob-bar-fill").forEach((bar) => {
        bar.style.width = bar.dataset.pct + "%";
      });
    });
  }
}

// ── API Call ────────────────────────────────────────────────────
/**
 * Collect form values, POST to Flask API, and handle the response.
 */
async function submitPrediction(event) {
  event.preventDefault();
  hideError();
  hideResult();

  // Validate
  const { valid, errors } = validateForm();
  if (!valid) {
    showError(errors.join("  •  "));
    return;
  }

  // Build payload
  const payload = {
    budget:  parseFloat(document.getElementById("budget").value),
    revenue: parseFloat(document.getElementById("revenue").value),
    rating:  parseFloat(document.getElementById("rating").value),
    votes:   parseFloat(document.getElementById("votes").value),
    runtime: parseFloat(document.getElementById("runtime").value),
    year:    parseInt(document.getElementById("year").value, 10),
    roi:     parseFloat(document.getElementById("roi").value),
    genre:   document.getElementById("genre").value,
  };

  setLoading(true);

  try {
    const response = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      // API returned an error (4xx / 5xx)
      showError(data.error || `API error: HTTP ${response.status}`);
      return;
    }

    if (!data.prediction) {
      showError("Unexpected API response — 'prediction' field missing.");
      return;
    }

    showResult(data.prediction, data.probabilities || null);

  } catch (err) {
    // Network failure or CORS issue
    if (err instanceof TypeError && err.message.includes("fetch")) {
      showError(
        "Cannot reach the API at " + API_URL +
        ". Make sure the Flask server is running: python app.py"
      );
    } else {
      showError("Unexpected error: " + err.message);
    }
  } finally {
    setLoading(false);
  }
}

// ── Clear `.invalid` class on user interaction ──────────────────
REQUIRED_FIELDS.forEach((fieldId) => {
  const el = document.getElementById(fieldId);
  if (el) {
    el.addEventListener("input",  () => el.classList.remove("invalid"));
    el.addEventListener("change", () => el.classList.remove("invalid"));
  }
});

// ── Event Listeners ─────────────────────────────────────────────
form.addEventListener("submit", submitPrediction);
