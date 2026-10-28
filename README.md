# Keystay — Booking Assurance for Hotels

**Predicting hotel booking cancellations to cut revenue loss.**
An AI system that flags risky reservations at booking time and turns each prediction into a revenue decision: a reminder, a deposit, an offer, or no action.

Built on **119,390 real hotel bookings** (Kaggle *Hotel booking demand*) and applied to the Indian hospitality context (OYO, Taj, MakeMyTrip).

> **Live demo:** `https://<your-username>.github.io/keystay/` (see *Deploy* below)

---

## Highlights

| | |
|---|---|
| Bookings analysed | 119,390 (Jul 2015 – Aug 2017) |
| Cancellation rate | 37.0% |
| Revenue lost to cancellations | ≈ ₹158.9 Cr (€16.73 M at €1 = ₹95) |
| Model | Logistic regression, 34 features, trained on 2015–16 and tested on unseen 2017 bookings |
| Test results (Phase 2) | **76.7%** accuracy, 76.4% precision, 57.5% recall |
| Revenue protected at threshold 0.50 (one season) | ≈ ₹12.17 Cr |

## Features

- **Login and sign-up** with password strength meter (demo authentication: accounts are stored only in the visitor's browser)
- **Refresh-safe routing**: reloading keeps you on the same page, and filters and sliders are remembered
- **Overview**: revenue lost, cancellation rate, monthly trend, with hotel / year / exchange-rate filters
- **Cancellation drivers**: lead time, segment, special requests, deposit type, guest history
- **Prediction model**: Phase 1 vs Phase 2 accuracy and precision; the full list of 12 and 34 features; interactive confusion matrix with a plain-language threshold explanation; coefficient chart
- **Channel scorecard**: bookings, cancellation rate, revenue lost and a Push / Monitor / Restrict verdict for every booking channel, so marketing knows which channels to grow and which to restrict
- **Booking risk scorer**: score any reservation live, with plain-language reasons and a recommended action
- **Revenue policy simulator**: revenue protected by a deposit/reminder policy at every risk threshold
- **Guest risk tiers**: every 2017 booking placed in Secure / Watch / At risk, with the actual cancellation rate and recommended action for each tier
- **Upload & score (add-on)**: upload any bookings CSV in the Kaggle format. Every row gets a cancellation probability, risk tier and recommended action, and you can download the scored CSV. If the file has `is_canceled`, the model is also evaluated on it. Try it with `data/sample_bookings.csv`.

All analysis, including model training, runs **in the browser** on the real dataset. There is no backend.

## Project structure

```
keystay/
├── index.html              # app shell: auth + 8 dashboard pages
├── styles.css              # design system (white linen, ink navy, brass accent)
├── app.js                  # auth, routing, charts, model training, simulators, CSV scoring
├── data.js                 # compact encoding of the dataset (generated)
├── data/
│   ├── hotel_bookings.csv  # original Kaggle dataset (119,390 × 32)
│   └── sample_bookings.csv # 500 real 2017 bookings to try the upload page
├── scripts/
│   └── prepare_data.py     # rebuilds data.js from the CSV (standard library only)
└── docs/
    └── PROJECT_REPORT.md   # full project report: problem, data, method, results
```

## Run locally

Open `index.html` in any modern browser. It needs an internet connection for fonts and Chart.js.

To regenerate `data.js` from the CSV:

```bash
python scripts/prepare_data.py data/hotel_bookings.csv data.js
```

## Deploy (GitHub Pages)

1. Push this folder to a public repository named `keystay`.
2. On GitHub, open **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**, then branch `main` and folder `/ (root)`, and click **Save**.
4. After about a minute the site is live at `https://<your-username>.github.io/keystay/`.

## Methodology in brief

1. **Leakage control:** removed `reservation_status`, `reservation_status_date`, `assigned_room_type` and `booking_changes`, which are only known after the outcome.
2. **Time-based validation:** train on 2015–16 arrivals (78,703), test on 2017 arrivals (40,687).
3. **Two-phase model:** booking basics only (63.6% accuracy), then guest history, channel and deposit added (76.7% accuracy). Guest nationality is deliberately not used, so guests are not treated differently by country.
4. **Decision layer:** risk tiers (Secure / Watch / At risk) and a revenue-protected simulator. GenAI guest messaging is documented in the project report.

See [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) for the full write-up.

## Dataset and citation

- Kaggle: [Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets.* **Data in Brief**, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

The data comes from two hotels in Portugal and is used here as a proxy for an Indian hotel chain. The website displays arrival years shifted forward by seven (shown as 2022–2024); the source data covers July 2015 – August 2017, and all figures are unchanged. Prices are in euros, converted at an editable rate.

## Disclaimer

Academic project. The revenue figures depend on an assumed recovery rate (default 30%). Sign-in is a front-end demo, so don't reuse a real password.
