# Supply Chain Delivery Performance Analysis

End-to-end analysis of an e-commerce supply chain dataset to understand **why deliveries run late**, **what delays cost the business**, and whether late deliveries can be **predicted before they happen**.

## Business Problem

A global e-commerce company selling sporting goods manages order fulfilment and shipping across multiple regions. Delivery performance is inconsistent: actual shipping times frequently deviate from the promised schedule, producing late deliveries and unpredictable order profitability.

**Goal:** analyze delivery operations, identify the bottlenecks behind late deliveries, quantify the profit impact, and build a machine learning model that flags orders at risk of arriving late.

## Dataset

| | |
|---|---|
| **File** | `DataCoSupplyChainDataset.csv` |
| **Rows × columns (raw)** | 180,519 × 53 |
| **After cleaning** | 172,765 × 20 |
| **Period covered** | January 2015 – February 2018 |
| **Grain** | One row per order item, with customer, product, shipping and payment attributes |

The raw file contained identifier, PII (customer names, emails, passwords), and redundant columns that were dropped — `Product Description` is empty in every row and `Order Zipcode` is missing in 86% of rows. Orders with `Shipping canceled` status were removed since they never enter the delivery pipeline. Two date fields were converted to datetime, and four analysis features were engineered:

- **Order Processing Time** – days between order placement and shipping date
- **Delay** – processing time minus scheduled shipping days (negative = early)
- **Is_Delayed** – whether the delay is positive
- **Profitability Flag** – Profit / Loss / Break-even per order

## Key Insights

### 1. Late delivery is the norm, not the exception

- **54.7% of orders ship late** (94,523 of 172,765); only 45.3% arrive on time or early
- The **most common outcome is exactly 1 day late — 31.0% of all orders**
- 24.1% of orders actually ship **early** (1–2 days ahead of schedule)
- 90% of delays are **3 days or fewer** — this is a systematic scheduling problem, not a long-tail logistics failure

### 2. The scheduling system is miscalibrated, especially for First Class

| Shipping Mode | Scheduled days | Actual behaviour | Delay rate |
|---|---|---|---|
| **First Class** | 1 day | Always takes 2 days | **100% late** |
| **Second Class** | ~2 days | Usually takes longer | **79.8% late** |
| **Standard Class** | 4 days | Often beats the clock | **39.8% late** |
| **Same Day** | 0 days | Meets the promise | **0% late** |

Every single First Class shipment (26,513 orders) is scheduled for 1 day and takes exactly 2. The premium ("fast") tiers consistently under-deliver on their promise, while Standard Class routinely ships early. Simply re-baselining First and Second Class schedules would eliminate most recorded lateness overnight — and would let the company set honest delivery promises at checkout.

### 3. Delays are not (yet) destroying margin — but loss-making orders are widespread

- Mean profit per order is roughly flat across delay levels (~$20–23), dipping only slightly for 3-day-late orders (~$20.0)
- Orders that shipped late still generated **~$2.1M in net profit**, so the cost of lateness shows up as a service/reliability problem rather than an immediate margin hit
- The bigger profitability issue: **18.7% of all orders (32,295) lose money outright**, i.e. sold below cost

### 4. Lateness is structural, not behavioral

Delay rates cluster in a narrow 53–55% band across almost every dimension:

| Dimension | Spread |
|---|---|
| Order status | PAYMENT_REVIEW 55.3% vs ON_HOLD 53.0% |
| Payment type | PAYMENT 55.0% vs CASH 53.8% |
| Customer segment | Home Office 55.4% vs Corporate 54.5% |
| Day of week | Monday 55.5% vs Wednesday 54.8% |
| Hour of day | 51.9% (7pm) to 57.1% (8pm) |

Who orders, how they pay, and when they order barely move the needle. The delay is baked into the **shipping-mode schedule itself** — so a global scheduling fix will outperform any targeted intervention at specific customer, payment, or time-of-day segments.

### 5. Regional and seasonal variation is real but narrow

- Worst region: **Central Africa (58.7%)**; best: Canada (50.8%, small sample)
- Monthly delay rate stays within 53.7% (Jul) to 55.4% (Aug/Sep), with a secondary bump in December (55.2%)
- A drill-down into Central Africa (the worst region) confirms the structural pattern: First Class shipping (100% late, n=278), Second Class (82.8%) and the Golf department (60.8%) top the driver list

## Machine Learning: Late Delivery Prediction

A Random Forest classifier predicts the dataset's `Late_delivery_risk` flag (1 = order will ship late) using only information available **at or before dispatch** — no leakage from actual shipping times.

**Features:** payment type, scheduled shipping days, product category, customer segment, department, order region, shipping mode, order month, order hour.

**Pipeline:**
1. Categorical features frequency-encoded (robust for high-cardinality columns)
2. 80/20 stratified train/test split (34,553 held-out test rows)
3. **SMOTE** oversampling on the training set only (79,182 vs 59,030 → balanced)
4. `RandomForestClassifier` (scikit-learn)

**Results on the held-out test set:**

| Metric | Score |
|---|---|
| Accuracy | **0.74** |
| Precision (late class) | **0.79** |
| Recall (late class) | **0.75** |
| F1 (late class) | **0.77** |
| Baseline (majority class) | 0.57 |

The model lifts accuracy ~17 points over naively guessing "late" every time, with precision weighted toward the business-critical class: when it flags an order as at-risk, it is right ~4 times out of 5. With 54.7% of orders genuinely running late, this gives operations a usable early-warning signal for proactive intervention.

## Repository Structure

```
Supply_Chain/
├── README.md                        # This file
├── Supply_Chain_Analysis.ipynb      # Full analysis: EDA → KPIs → ML
└── DataCoSupplyChainDataset.csv     # Source data
```

## How to Run

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Supply_Chain_Analysis.ipynb
   ```
4. Run all cells — the CSV is read from the project root

## Tools & Techniques

**Python** · pandas · NumPy · matplotlib · seaborn · scikit-learn · imbalanced-learn (SMOTE)

EDA · KPI definition · delay & profitability feature engineering · bottleneck analysis by region / mode / status / department · time-based seasonality analysis · class-imbalance handling · Random Forest classification

## Recommendations

1. **Re-baseline First and Second Class delivery promises** — the single highest-impact fix, converting a 100%/80% lateness problem into honest scheduling
2. **Use the model at order entry** to set honest delivery estimates at checkout and route at-risk orders into a faster fulfilment path proactively
3. **Review pricing on the 18.7% of orders sold at a loss** — discounting and shipping cost erosion is a larger profit leak than the delays themselves
4. **Skip micro-targeting by segment, payment type, or time of day** — the data shows delay rates are uniform across them; global scheduling fixes beat targeted programs

## Future Work

- Feature importance / SHAP analysis to make the model's drivers transparent
- Gradient boosting (XGBoost / LightGBM) benchmark and probability calibration for risk scoring
- Separate "will be late" from "how many days late" (regression head) to prioritise the worst cases
- Profit-aware model: predict expected profit impact, not just lateness
