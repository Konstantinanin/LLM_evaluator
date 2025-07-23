# LLM_evaluator
#  LLM Evaluation Framework

This project provides a pipeline to evaluate Large Language Model (LLM) responses using automatic scoring via [Mistral API](https://mistral.ai/). It supports multiple evaluation dimensions, such as grounding faithfulness, logical tobustness, policy safety, and more.

---

##  Setup

1. **Clone the repo** and install dependencies:
   ```bash
   pip install -r requirements.txt

2. **Set up API keys for Mistral**:
Create a .env file with the following content:

MISTRAL_API_KEY=your_api_key_here

3. **Prepare the CSV file with all the columns**

**To evaluate a dataset**:

python main.py --csv path/to/your_dataset.csv --temperature 0.0 --seed 42

--temperature: Controls randomness in the model's scoring (default: 0.0) - we need just scores in numbers (1 -5).

--seed: Set to 42 for reproducibility.

**Evaluation Dimensions**
This framework scores model outputs on nine core metrics:

Metric	Description
grounding faithfulness:	How much of the answer matches the retrieved facts
completeness: How much of the expected information is included - regardless correctness
language_appropriateness:	Is the tone, vocabulary, and structure suitable and clear?
contradiction: Does the answer contradict the retrieved sources?
policy_safety:	Is it aligned with ethical & safety policies?
task_completion:	Did the assistant do what was asked?
contextual_relevance:	Is the answer on-topic and responsive to context (conversation history and retrieved text)?
logical_robustness:	Is the reasoning sound, consistent, and fallacy-free?

Each metric is scored from 1 (poor) to 5 (excellent).

**Output will be saved to**:

reports/graded_<timestamp>.csv

reports/report_<timestamp>.md

**Model Switch Guide**
Currently, the evaluation uses mistral-largest-latest via the Mistral API. To change the model:

Modify the self.model line in MetricScorer.__init__().
self.model = "mistral-medium-2505" 

**Output**
Each evaluation run produces:

A .csv with per-sample scores for all metrics

A .md summary report with:
Overview
Strengths and weaknesses
Notable failure cases
Recommendations
