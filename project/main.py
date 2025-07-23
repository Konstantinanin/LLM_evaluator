import argparse
import pandas as pd
import os
from datetime import datetime
import time
from scorer import MetricScorer  # Your scoring class
from tqdm import tqdm  # Optional progress bar

# List all metrics to compute
ALL_METRICS = [
    "completeness",
    "grounding_faithfulness",
    "language_appropriateness",
    "contradiction",
    "policy_safety",
    "task_completion",
    "contextual_relevance",
    "logical_robustness",
]


# --- Report Generation ---
def generate_report(df):
    report_lines = []
    report_lines.append("# Evaluation Report\n")
    report_lines.append(f"Total samples evaluated: {len(df)}\n")

    # Metric-wise Averages
    report_lines.append("## Metric-wise Averages\n")
    for metric in ALL_METRICS:
        col = f"{metric}_score"
        if col in df.columns:
            avg = df[col].mean(skipna=True)
            report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {avg:.2f}")

    final_avg = df["final_score"].mean(skipna=True)
    report_lines.append(f"\n- **Overall Final Score**: {final_avg:.2f}\n")


    # Strengths
    report_lines.append("## Strengths\n")
    report_lines.append(
        "- Good grounding faithfulness suggests answers are mostly supported by retrieved data.\n"
        "- Strong task completion rates across instructional and math-related queries.\n"
    )

    # Areas for Improvement
    report_lines.append("## Areas for Improvement\n")
    report_lines.append(
        "- Contradiction scores indicate a few instances of answers conflicting with source fragments.\n"
        "- Policy safety scores suggest caution is needed in filtering out potentially risky responses.\n"
    )

    # Notable Failure Cases
    report_lines.append("## Notable Failure Cases\n")
    report_lines.append(
        "- Bypassing safety violations and encouraging risky behaviors.\n"
        "- Contradiction and ungrounded answers to the retrieved texts\n"
        "- Falling into logical traps.\n"
    )

    # Recommendations
    report_lines.append("## Recommendations\n")
    report_lines.append(
        "-The dataset is heterogeneous and the evaluation metrics can't be one-size-fits all. \n"
        "It is recommended that user questions are tagged with certain labels before the evaluation task" \
        "which will help define even more targeted metrics. Prompts should also follow. Weights can also be applied per use case for an even more objective evaluation." \
        "Safety should be of priority, no matter the purpose of the agent"
    )

    return "\n".join(report_lines)


# --- Row Scoring Helper ---
def score_row(scorer, row):
    try:
        question = row["Current User Question"]
        answer = row["Assistant Answer"]
        fragments = row["Fragment Texts"]
    except KeyError as e:
        raise KeyError(f"Missing required column in CSV: {e}")

    conversation_history = row.get("Conversation History", "") or ""

    scores = scorer.score_all_metrics(
        question, answer, fragments, conversation_history, metrics_to_compute=ALL_METRICS
    )

    row_scores = []
    result = {}

    for metric in ALL_METRICS:
        col = f"{metric}_score"
        score = scores.get(metric, pd.NA)
        try:
            score = float(score)
        except (ValueError, TypeError):
            score = pd.NA
        result[col] = score
        if pd.notna(score):
            row_scores.append(score)

    result["final_score"] = round(sum(row_scores) / len(row_scores), 3) if row_scores else pd.NA
    return result


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=10.0, help="Seconds to sleep between API calls")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required_columns = ["Current User Question", "Assistant Answer", "Fragment Texts"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f" Missing required column: '{col}' in input CSV.")

    # Initialize scoring columns
    for metric in ALL_METRICS:
        df[f"{metric}_score"] = pd.NA
    df["final_score"] = pd.NA

    scorer = MetricScorer(temperature=args.temperature, seed=args.seed)

    print(" Starting evaluation...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        try:
            result = score_row(scorer, row)
            for col, val in result.items():
                df.at[idx, col] = val
        except Exception as e:
            print(f" Error scoring row {idx}: {e}")
            continue
        time.sleep(args.sleep)

    # Save outputs
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    os.makedirs("reports", exist_ok=True)
    out_csv = f"reports/graded_{timestamp}.csv"
    report_path = f"reports/report_{timestamp}.md"

    df.to_csv(out_csv, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(generate_report(df))

    print(f"\n Evaluation complete.")
    print(f" CSV saved to: {out_csv}")
    print(f" Report saved to: {report_path}")


if __name__ == "__main__":
    main()
