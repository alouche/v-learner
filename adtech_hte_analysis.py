"""
Learners compared:
- S-Learner (RF)
- T-Learner (RF)
- X-Learner (RF)
- V-Learner (DR + cross-fitting + variance weighting + bootstrap uncertainty)
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from econml.metalearners import XLearner, TLearner, SLearner
from v_learner import VLearner


def generate_adtech_data(n_samples=5000, random_state=42):
    """Generate synthetic AdTech data with known heterogeneous treatment effects."""
    np.random.seed(random_state)

    age = np.random.normal(35, 12, n_samples).clip(18, 70)
    income_segment = np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2])

    time_on_site = np.random.exponential(180, n_samples).clip(10, 1800)
    pages_viewed = np.random.poisson(4, n_samples).clip(1, 20)
    previous_purchases = np.random.poisson(2, n_samples).clip(0, 20)

    device_mobile = np.random.binomial(1, 0.6, n_samples)
    hour_of_day = np.random.randint(0, 24, n_samples)
    is_weekend = np.random.binomial(1, 2 / 7, n_samples)

    session_count = np.random.poisson(5, n_samples).clip(1, 50)
    days_since_last_visit = np.random.exponential(7, n_samples).clip(0, 90)

    X = np.column_stack(
        [
            (age - 35) / 12,
            income_segment / 3,
            np.log1p(time_on_site) / 10,
            np.log1p(pages_viewed) / 5,
            np.log1p(previous_purchases) / 5,
            device_mobile,
            np.sin(2 * np.pi * hour_of_day / 24),
            np.cos(2 * np.pi * hour_of_day / 24),
            is_weekend,
            np.log1p(session_count) / 5,
            np.log1p(days_since_last_visit) / 5,
        ]
    )

    feature_names = [
        "age_normalized",
        "income_segment",
        "log_time_on_site",
        "log_pages_viewed",
        "log_previous_purchases",
        "device_mobile",
        "hour_sin",
        "hour_cos",
        "is_weekend",
        "log_session_count",
        "log_days_since_last_visit",
    ]

    user_quality_score = 0.3 * X[:, 1] + 0.2 * X[:, 4] + 0.2 * X[:, 9] - 0.1 * X[:, 10]
    propensity_logit = user_quality_score + np.random.normal(0, 0.5, n_samples)
    propensity = 1 / (1 + np.exp(-propensity_logit))
    T = np.random.binomial(1, propensity)

    baseline_logit = (
        -3.0
        + 0.3 * X[:, 1]
        + 0.2 * X[:, 4]
        + 0.15 * X[:, 3]
        + 0.1 * X[:, 2]
        - 0.2 * X[:, 10]
    )

    true_cate_logit = (
        0.5
        + 0.4 * X[:, 1]
        + 0.3 * (X[:, 0] ** 2)
        + 0.25 * X[:, 3] * X[:, 2]
        + 0.3 * X[:, 5]
        - 0.2 * X[:, 10]
        + 0.15 * X[:, 4]
    )

    prob_baseline = 1 / (1 + np.exp(-baseline_logit))
    prob_treated = 1 / (1 + np.exp(-(baseline_logit + true_cate_logit)))
    true_cate = np.clip(prob_treated - prob_baseline, -0.5, 0.5)

    outcome_logit = baseline_logit + T * true_cate_logit
    conversion_prob = 1 / (1 + np.exp(-outcome_logit))
    Y = np.random.binomial(1, conversion_prob)

    df = pd.DataFrame(X, columns=feature_names)
    df["treatment"] = T
    df["conversion"] = Y
    df["true_cate"] = true_cate

    df["age"] = age
    df["income_segment_raw"] = income_segment
    df["time_on_site"] = time_on_site
    df["pages_viewed"] = pages_viewed
    df["previous_purchases"] = previous_purchases
    df["device_mobile_raw"] = device_mobile
    df["hour_of_day"] = hour_of_day
    df["is_weekend_raw"] = is_weekend
    df["session_count"] = session_count
    df["days_since_last_visit"] = days_since_last_visit

    print("Dataset Summary:")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Treatment rate: {T.mean():.1%}")
    print(f"  Overall conversion rate: {Y.mean():.2%}")
    print(f"  Treated conversion rate: {Y[T == 1].mean():.2%}")
    print(f"  Control conversion rate: {Y[T == 0].mean():.2%}")
    print(f"  Naive ATE: {(Y[T == 1].mean() - Y[T == 0].mean()):.2%}")
    print(f"  True CATE: mean={true_cate.mean():.2%}, std={true_cate.std():.2%}")
    print(f"  True CATE range: [{true_cate.min():.2%}, {true_cate.max():.2%}]")

    return df, feature_names


def train_and_evaluate_learners(df, feature_names):
    print("TRAINING META-LEARNERS FOR HTE ESTIMATION")

    X = df[feature_names].values
    T = df["treatment"].values
    Y = df["conversion"].values
    true_cate = df["true_cate"].values

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=T
    )

    X_train, X_test = X[train_idx], X[test_idx]
    T_train, T_test = T[train_idx], T[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    true_cate_test = true_cate[test_idx]

    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    base_rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )

    cate_et = ExtraTreesRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    learners = {
        "S-Learner (RF)": SLearner(overall_model=base_rf),
        "T-Learner (RF)": TLearner(models=base_rf),
        "X-Learner (RF)": XLearner(
            models=base_rf,
            propensity_model=RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            ),
            cate_models=base_rf,
        ),
        "V-Learner (DR+CF)": VLearner(
            propensity_model=RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1
            ),
            outcome_model=None,
            cate_model=cate_et,
            n_splits=5,
            n_bootstrap=50,
            propensity_clip=(0.02, 0.98),
            pseudo_outcome_clip_quantiles=(0.01, 0.99),
            use_variance_stabilizing_weights=True,
            random_state=42,
        ),
    }

    results = []
    predictions = {}
    vlearner_obj = None

    import time

    for name, learner in learners.items():
        start = time.time()
        learner.fit(Y_train, T_train, X=X_train)
        fit_time = time.time() - start

        start = time.time()
        cate_pred = learner.effect(X_test)
        pred_time = time.time() - start

        rmse = float(np.sqrt(mean_squared_error(true_cate_test, cate_pred)))
        mae = float(mean_absolute_error(true_cate_test, cate_pred))
        r2 = float(
            1
            - np.sum((true_cate_test - cate_pred) ** 2)
            / np.sum((true_cate_test - true_cate_test.mean()) ** 2)
        )
        corr = float(np.corrcoef(true_cate_test, cate_pred)[0, 1])

        p25 = np.percentile(cate_pred, 25)
        p75 = np.percentile(cate_pred, 75)
        top = cate_pred >= p75
        mid = (cate_pred >= p25) & (cate_pred < p75)
        bot = cate_pred < p25
        top_true = float(true_cate_test[top].mean())
        mid_true = float(true_cate_test[mid].mean())
        bot_true = float(true_cate_test[bot].mean())

        results.append(
            {
                "Learner": name,
                "RMSE": rmse,
                "MAE": mae,
                "R²": r2,
                "Correlation": corr,
                "Top25 True CATE": top_true,
                "Mid50 True CATE": mid_true,
                "Bot25 True CATE": bot_true,
                "Train Time": fit_time,
                "Predict Time": pred_time,
            }
        )
        predictions[name] = cate_pred

        print(f"Training complete: {name}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Correlation: {corr:.4f}")
        print("  Segment true-CATE means (test):")
        print(f"    TOP25 true CATE: {top_true:.4f}")
        print(f"    MID50 true CATE: {mid_true:.4f}")
        print(f"    BOT25 true CATE: {bot_true:.4f}")
        print(f"    Overall true CATE: {true_cate_test.mean():.4f}")

        if "V-Learner" in name:
            vlearner_obj = learner

    results_df = pd.DataFrame(results)

    print("BENCHMARK RESULTS SUMMARY")
    print(results_df.to_string(index=False))

    return results_df, predictions, true_cate_test, df, vlearner_obj


def analyze_business_insights(df, vlearner, feature_names):
    print("BUSINESS INSIGHTS & RECOMMENDATIONS")

    X = df[feature_names].values

    cate_pred = np.clip(vlearner.effect(X), -0.5, 0.5)
    uncertainty = vlearner.effect_uncertainty(X)

    df_insights = df.copy()
    df_insights["predicted_lift"] = cate_pred * 100
    df_insights["uncertainty"] = uncertainty

    print("Overall Campaign Performance:")
    print(f" Average predicted lift: {cate_pred.mean() * 100:.2f}%")
    print(f" Lift range: [{cate_pred.min() * 100:.2f}%, {cate_pred.max() * 100:.2f}%]")
    print(
        f" Users with positive lift: {(cate_pred > 0).sum():,} ({(cate_pred > 0).mean():.1%})"
    )

    print("Targeting Recommendations:")

    high_lift_threshold = np.percentile(cate_pred, 75)
    high_lift_mask = cate_pred >= high_lift_threshold
    print("1. HIGH-VALUE SEGMENT (Top 25% predicted lift)")
    print(f"   Size: {high_lift_mask.sum():,} users ({high_lift_mask.mean():.1%})")
    print(f"   Avg predicted lift: {cate_pred[high_lift_mask].mean() * 100:.2f}%")
    print("   Characteristics:")
    print(
        f"     - Income: {df_insights[high_lift_mask]['income_segment_raw'].mean():.2f} (0-3 scale)"
    )
    print(f"     - Avg age: {df_insights[high_lift_mask]['age'].mean():.1f} years")
    print(
        f"     - Previous purchases: {df_insights[high_lift_mask]['previous_purchases'].mean():.1f}"
    )
    print(
        f"     - Mobile users: {df_insights[high_lift_mask]['device_mobile_raw'].mean():.1%}"
    )

    medium_lift_mask = (cate_pred >= np.percentile(cate_pred, 25)) & (
        cate_pred < high_lift_threshold
    )
    print("2. MEDIUM-VALUE SEGMENT (25-75th percentile)")
    print(f"   Size: {medium_lift_mask.sum():,} users ({medium_lift_mask.mean():.1%})")
    print(f"   Avg predicted lift: {cate_pred[medium_lift_mask].mean() * 100:.2f}%")
    print("   Recommendation: Test with A/B experiment")

    low_lift_mask = cate_pred < np.percentile(cate_pred, 25)
    print("3. LOW-VALUE SEGMENT (Bottom 25%)")
    print(f"   Size: {low_lift_mask.sum():,} users ({low_lift_mask.mean():.1%})")
    print(f"   Avg predicted lift: {cate_pred[low_lift_mask].mean() * 100:.2f}%")
    print("   Recommendation: Save ad spend, use standard placements")

    print("UNCERTAINTY ANALYSIS:")
    high_uncertainty_mask = uncertainty > np.percentile(uncertainty, 90)
    print(
        f" High-uncertainty predictions: {high_uncertainty_mask.sum():,} users ({high_uncertainty_mask.mean():.1%})"
    )
    print("  Recommendation: Collect more data or run experiments for these users")

    print("ROI PROJECTION:")
    premium_ad_cost = 0.50
    conversion_value = 25.00

    expected_incremental_conversions = np.sum(cate_pred[high_lift_mask])
    expected_revenue = expected_incremental_conversions * conversion_value
    expected_cost = high_lift_mask.sum() * premium_ad_cost
    expected_profit = expected_revenue - expected_cost
    roi = (expected_profit / expected_cost) * 100 if expected_cost != 0 else np.nan

    print(" If targeting HIGH-VALUE segment only:")
    print(f"   Ad spend: ${expected_cost:,.0f}")
    print(f"   Expected incremental revenue: ${expected_revenue:,.0f}")
    print(f"   Expected profit: ${expected_profit:,.0f}")
    print(f"   ROI: {roi:.1f}%")

    return df_insights


def create_visualizations(results_df, predictions, true_cate_test, df_insights):
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "AdTech HTE Demo: Meta-Learner Comparison", fontsize=16, fontweight="bold"
    )

    metrics = ["RMSE", "MAE", "R²", "Correlation"]
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        data = results_df.sort_values(metric, ascending=(metric in ["RMSE", "MAE"]))
        ax.barh(data["Learner"], data[metric])
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f"{metric} Comparison", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "outputs/adtech_performance_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Predicted vs True CATE", fontsize=16, fontweight="bold")

    for idx, (name, pred) in enumerate(list(predictions.items())[:4]):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(true_cate_test * 100, pred * 100, alpha=0.5, s=20)

        min_val = min(true_cate_test.min(), pred.min()) * 100
        max_val = max(true_cate_test.max(), pred.max()) * 100
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect"
        )

        r2 = 1 - np.sum((true_cate_test - pred) ** 2) / np.sum(
            (true_cate_test - true_cate_test.mean()) ** 2
        )
        ax.set_xlabel("True Conversion Lift (%)", fontsize=11)
        ax.set_ylabel("Predicted Conversion Lift (%)", fontsize=11)
        ax.set_title(f"{name} (R² = {r2:.3f})", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "outputs/adtech_predictions_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Business Insights: User Segmentation", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(df_insights["predicted_lift"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        df_insights["predicted_lift"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean",
    )
    ax.set_xlabel("Predicted Conversion Lift (%)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Predicted Lift", fontsize=12, fontweight="bold")
    ax.legend()

    ax = axes[0, 1]
    df_insights.groupby("income_segment_raw")["predicted_lift"].mean().plot(
        kind="bar", ax=ax
    )
    ax.set_xlabel("Income Segment (0=Low, 3=High)", fontsize=11)
    ax.set_ylabel("Avg Predicted Lift (%)", fontsize=11)
    ax.set_title("Lift by Income Segment", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)

    ax = axes[0, 2]
    age_bins = [18, 25, 35, 45, 55, 70]
    df_insights["age_group"] = pd.cut(df_insights["age"], bins=age_bins)
    df_insights.groupby("age_group")["predicted_lift"].mean().plot(kind="bar", ax=ax)
    ax.set_xlabel("Age Group", fontsize=11)
    ax.set_ylabel("Avg Predicted Lift (%)", fontsize=11)
    ax.set_title("Lift by Age Group", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1, 0]
    df_insights.groupby("device_mobile_raw")["predicted_lift"].mean().plot(
        kind="bar", ax=ax
    )
    ax.set_xlabel("Device (0=Desktop, 1=Mobile)", fontsize=11)
    ax.set_ylabel("Avg Predicted Lift (%)", fontsize=11)
    ax.set_title("Lift by Device Type", fontsize=12, fontweight="bold")
    ax.set_xticklabels(["Desktop", "Mobile"], rotation=0)

    ax = axes[1, 1]
    ax.scatter(
        df_insights["previous_purchases"],
        df_insights["predicted_lift"],
        alpha=0.3,
        s=10,
    )
    ax.set_xlabel("Previous Purchases", fontsize=11)
    ax.set_ylabel("Predicted Lift (%)", fontsize=11)
    ax.set_title("Lift vs Customer Loyalty", fontsize=12, fontweight="bold")

    ax = axes[1, 2]
    ax.hist(df_insights["uncertainty"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Bootstrap Disagreement (std)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Uncertainty Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig("outputs/adtech_business_insights.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("Heterogeneous Treatment Effect Estimation for Ad Campaign Optimization")

    df, feature_names = generate_adtech_data(n_samples=5000)

    results_df, predictions, true_cate_test, df_full, vlearner = (
        train_and_evaluate_learners(df, feature_names)
    )

    df_insights = analyze_business_insights(df_full, vlearner, feature_names)

    from pathlib import Path as _Path

    _Path("outputs").mkdir(exist_ok=True)

    create_visualizations(results_df, predictions, true_cate_test, df_insights)

    results_df.to_csv("outputs/adtech_benchmark_results.csv", index=False)
    df_insights.to_csv("outputs/adtech_predictions_with_insights.csv", index=False)


if __name__ == "__main__":
    main()
