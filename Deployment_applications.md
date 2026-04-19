# FinLangNet — Deployment & Real-World Applications

This document describes the production deployment of FinLangNet and the results from controlled online testing. It corresponds to **Section 5.3 (Online A/B Test)** and **Appendix D (Real-World Deployment Analysis)** in the paper.

---

## 1. Application Scenario

The credit management lifecycle can be divided into three stages:

- **Pre-loan**: admission review, credit line allocation, disbursement approval
- **During-loan**: credit line management, churn prediction, marketing response
- **Post-loan**: collections, repayment rate prediction, aging roll rate, lost contact prediction

FinLangNet is deployed for **Behavior Scoring (B-Score) during the mid-loan phase** in a cash loan service. The B-Score dynamically monitors risk fluctuations after disbursement. Offline computation on a T+1 basis is sufficient for this scenario, avoiding the need for real-time online serving. This scenario was selected because existing customers provide ample feature sources — sufficient loan application and repayment records — that facilitate behavior-based modeling.

---

## 2. Deployment Architecture

### 2.1 System Design

In real-world financial risk management, **regulatory compliance bans the use of black-box models for direct credit decisioning**. To bridge the gap between deep learning performance and industrial explainability requirements, FinLangNet is deployed as part of a **hybrid fusion framework**:

```
Raw Data (Inquiry Records, Account Ledgers, Behavioral Logs, Basic Info)
    │
    ├──► FinLangNet (Non-Sequential + Sequential Module)
    │         │
    │         └──► s_lang  (scalar language-risk sub-score)
    │
    └──► XGBoost (500+ manually engineered features)
              │
              └──► integrates s_lang as an additional dense feature
                        │
                        └──► Final Credit Decision (interpretable XGBoost output)
```

FinLangNet abstracts heterogeneous data streams into a single scalar sub-score `s_lang`. This score is fed into the interpretability-centric XGBoost system as a dense feature, effectively enhancing discriminative power without violating risk governance protocols.

### 2.2 Inference Pipeline

- **Computation frequency**: Offline, daily (T+1 basis)
- **Scope**: Full active customer base — millions of users per day
- **Infrastructure**: L20 GPU clusters
- **Throughput**: 100 QPS with sub-100ms latency per inference call
- **Monitoring**:
  - Daily: PSI (Population Stability Index) monitoring of features and output scores
  - Weekly: Vintage curve monitoring and bin delinquency rate tracking

### 2.3 Training and Integration Steps

1. **Daily batch inference**: FinLangNet generates the `s_lang` sub-score for all active users. Model outputs are stored for downstream utilization.

2. **Blended tree model training**: The deep learning sub-score `s_lang` is combined with the original XGBoost features to train the blended model. Apart from adding the sub-score, all XGBoost parameters remain unchanged. The resulting model consistently ranks `s_lang` highest among features by both IV value and feature importance score.

3. **Ongoing monitoring**: After deployment, model health is tracked via vintage curves, bin delinquency rates (weekly), and PSI for both features and output scores (daily).

---

## 3. Online A/B Test (Section 5.3)

### 3.1 Experimental Setup

A **controlled A/B test** was conducted at a representative **60% approval threshold** on the production platform. The two models compared were:

| Model | Description |
|---|---|
| **Control** (XGBoost) | Production model: 500+ manually engineered features, trained on domain expertise, optimized over years of deployment |
| **Treatment** (XGBoost + FinLangNet) | Adds FinLangNet's `s_lang` sub-score as an additional feature in XGBoost |

Both models were evaluated on the same primary business target: **y₁ (τ = 1), 30-day delinquency**.

### 3.2 Key Results

| Metric | Value |
|---|---|
| Absolute KS improvement | **+6.3 pp** |
| Relative reduction in bad debt rate | **−9.9%** |
| Default rate (Control at 60% approval) | ~9% |
| Default rate (Treatment at 60% approval) | ~8.2% |

The proposed framework **reduced the default rate from 9% to 8.2%** — a 9.9% relative improvement. This confirms that FinLangNet effectively discriminates high-risk applicants that traditional models miss.

### 3.3 Analysis

Further analysis with domain features yields a **+6.3 pp improvement in KS** (Table 1 in the paper), demonstrating the superiority of the multi-source temporal modeling approach. Combining sequential features with domain features yields a **12.35% reduction in expected losses**, validating the financial impact of the hybrid framework.

---

## 4. Swap Set Analysis

The swap set analysis tracks cases where the two models **disagree on risk classification** and measures actual delinquency outcomes to quantify the business impact of switching from XGBoost to XGBoost + FinLangNet.

### 4.1 Swap Matrix

| XGBoost \ XGBoost+FinLangNet | (-inf, 472.0] | (472.0, 569.0] | (569.0, 651.0] | (651.0, 756.0] | (756.0, inf] | Overdue Samples | Total Samples | % Samples | Overdue Rate | Pass Rate | Overdue / Passing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **(-inf, 473.0]**   | 39.16% | 22.95% | 16.41% | 12.98% | 0.00%  | 232,438 | 658,086 | 20.00% | 35.32% | 100.00% | 16.11% |
| **(473.0, 557.0]**  | 27.49% | 19.54% | 14.50% | 10.38% | 8.10%  | 130,362 | 659,658 | 20.05% | 19.76% | 80.00%  | 11.30% |
| **(557.0, 634.0]**  | 24.35% | 17.39% | 12.82% | 9.18%  | 5.51%  | 88,430  | 663,629 | 20.17% | 13.33% | 59.95%  | 8.47%  |
| **(634.0, 731.0]**  | 20.92% | 15.39% | 11.41% | 7.78%  | 4.87%  | 54,188  | 653,209 | 19.85% | 8.30%  | 39.78%  | 6.01%  |
| **(731.0, inf]**    | 33.33% | 12.54% | 9.71%  | 6.30%  | 3.16%  | 24,542  | 655,703 | 19.93% | 3.74%  | 19.93%  | 3.74%  |
| **Overdue Samples** | 240,396 | 129,771 | 85,344 | 51,780 | 22,669 | — | — | — | — | — | — |
| **Total Samples**   | 661,833 | 659,743 | 658,462 | 656,056 | 654,191 | — | — | — | — | — | — |
| **% Samples**       | 20.11%  | 20.05%  | 20.01%  | 19.94%  | 19.88%  | — | — | — | — | — | — |
| **Overdue Rate**    | 36.32%  | 19.67%  | 12.96%  | 7.89%   | 3.47%   | — | — | — | — | — | — |
| **Pass Rate**       | 100.00% | 79.89%  | 59.83%  | 39.82%  | 19.88%  | — | — | — | — | — | — |
| **Overdue / Passing** | 16.11% | 11.02% | 8.12% | 5.68% | 3.47% | — | — | — | — | — | — |
| **Relative Δ Overdue** | **+2.84%** | | | | **−7.42%** | | | | | | |

> **Relative Δ Overdue = (Overdue Rate[XGB+FLN] − Overdue Rate[XGB]) / Overdue Rate[XGB]**

### 4.2 Binning Comparison

**XGBoost**: (−∞, 473.0] | (473.0, 557.0] | (557.0, 634.0] | (634.0, 731.0] | (731.0, +∞)

**XGBoost + FinLangNet**: (−∞, 472.0] | (472.0, 569.0] | (569.0, 651.0] | (651.0, 756.0] | (756.0, +∞)

The shift in bin boundaries reflects how FinLangNet alters the model's perception of the risk distribution.

### 4.3 Swap Set Interpretation

**Bottom bin (−∞, 472.0] — +2.84% relative increase in overdue rate:**
The XGBoost + FinLangNet model identifies more genuine high-risk customers in this segment. This enhanced sensitivity to potential defaulters is critical for early risk detection and prevention of bad debt.

**Top bin (756.0, +∞) — −7.42% relative decrease in overdue rate:**
The new model achieves more accurate identification of truly low-risk customers in the high-score segment. This improvement in precision for creditworthy customers enables financial institutions to offer better loan conditions to high-quality clients without increasing portfolio risk.

---

## 5. Real-World Deployment Analysis (Appendix D)

### 5.1 Experimental Setup

Two modeling paradigms were compared on a held-out one-month test window following the training period, evaluated on the primary target y₁ (τ = 1, 30-day delinquency):

| Model | Description |
|---|---|
| **XGBoost (Baseline)** | Production model utilizing 500+ manually engineered features derived from domain expertise, optimized over years of deployment |
| **FinLangNet** | Proposed model processing raw sequential data without manual feature engineering, trained with multi-task learning across 7 risk-related objectives |

### 5.2 Threshold-Based Performance Analysis

Risk control systems require careful threshold selection to balance risk exposure (false negatives) and customer experience (false positives). Analysis of the precision-recall trade-off at various decision thresholds reveals:

- **Low Thresholds (0.0–0.2)**: XGBoost achieves higher recall but suffers from poor precision, creating operational challenges with excessive false positives.

- **Operational Range (0.2–0.4)**: FinLangNet maintains balanced performance with significantly higher precision while preserving competitive recall, reducing manual review costs at the thresholds commonly used in production.

- **High Thresholds (0.4+)**: Both models converge in performance, though FinLangNet maintains a slight precision advantage.

**Key finding**: FinLangNet's false positives tend to cluster in the moderate-risk range (0.3–0.5), making them easier to identify and handle through secondary screening — reducing operational overhead compared to XGBoost's false positives which are distributed more broadly.

### 5.3 Risk Distribution Analysis

Analysis of predicted score distributions versus actual default labels reveals fundamental differences in how the models discriminate risk:

- **Risk Separation**: FinLangNet produces clearer separation between defaulters and non-defaulters, especially in the high-risk segment (predicted scores > 0.6). The bimodal distribution achieved by FinLangNet reflects a sharper decision boundary.

- **Score Calibration**: XGBoost shows some score clustering around certain values (likely due to dominant features), while FinLangNet provides a more continuous and uniform risk scoring across the full score range.

These distributional properties reflect FinLangNet's ability to capture nuanced behavioral signals from temporal sequences that static feature-based models cannot encode.

### 5.4 Operational Impact Analysis

To assess real-world deployment benefits, swap set analysis was conducted — tracking cases where the two models disagree on risk classification and measuring actual delinquency outcomes:

| Finding | Detail |
|---|---|
| **High-Risk Captures** | Among users classified as high-risk only by FinLangNet, **68% showed delinquent behavior within 60 days**, validating its superior risk detection |
| **False Positive Reduction** | FinLangNet correctly identified **42% of XGBoost's false positives** as low-risk, potentially reducing unnecessary credit restrictions |
| **Portfolio Performance** | Applying FinLangNet's risk scores would reduce the portfolio default rate by an estimated **12.3%** while maintaining the same approval rate |

**Hybrid strategy**: Recognizing the complementary strengths of both approaches, the deployed system incorporates FinLangNet's representations as additional features in the XGBoost framework. This integration leverages FinLangNet's automated feature learning from raw sequential data alongside XGBoost's domain-specific features.

The hybrid model achieved a **6.3 pp improvement in KS metric** over standalone XGBoost, providing significant business value while maintaining full regulatory compliance through the interpretable XGBoost decision layer.

---

## 6. Conclusions and Recommendations

### 6.1 Enhancing Precision in High-Risk Identification

With improved predictive performance in the lower score segment (−∞, 472.0], financial institutions should leverage this advantage by:
- Adjusting credit policies for the bottom-score segment
- Reinforcing risk control measures to reduce bad loan rates
- Using FinLangNet's temporal signals for early warning triggers

### 6.2 Optimizing Services for Low-Risk Customers

The decrease in delinquency rates in the high-score segment (756.0, +∞) indicates that the new model better distinguishes genuinely low-risk customers. Financial institutions can:
- Offer more favorable loan conditions to customers in this segment
- Increase marketing efforts to attract high-quality clients
- Expand credit lines selectively for this group

### 6.3 Continuous Monitoring and Model Refinement

The deployed FinLangNet+XGBoost model demonstrates strong initial performance, but ongoing management is essential:
- **Daily**: PSI monitoring of feature distributions and output scores to detect distribution shift
- **Weekly**: Vintage curve and bin delinquency rate tracking for long-term stability assessment
- **Periodic**: Retrain with updated data to maintain model freshness as customer behavior evolves
- **Quarterly**: Full model re-evaluation and potential threshold recalibration based on current portfolio composition
