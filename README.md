# LLM evaluator


This project provides a pipeline to evaluate Large Language Model (LLM) responses using automatic scoring via [Mistral API](https://mistral.ai/). It supports multiple evaluation dimensions, such as grounding faithfulness, logical robustness, policy safety, and more.

---

##  Setup

1. **Clone the repo** and install dependencies:
   ```bash
   pip install -r requirements.txt

2. **Set up API keys for Mistral**:
Create a .env file with the following content:

MISTRAL_API_KEY=your_api_key_here

3. **Prepare the CSV file with all the columns**

**To evaluate the dataset**:

python main.py --csv path/to/rag_evaluation_07_2025.csv --temperature 0.0 --seed 42

--temperature: Controls randomness in the model's scoring (default: 0.0) - we need just scores in numbers (1 -5).

--seed: Set to 42 for reproducibility.

**Evaluation Dimensions**


This framework scores model outputs on 8 core metrics:



1. grounding faithfulness:	How much of the answer matches the retrieved facts

2. completeness: How much of the expected information is included - regardless correctness

3. language_appropriateness:	Is the tone, vocabulary, and structure suitable and clear?

4. contradiction: Does the answer contradict the retrieved sources?

5. policy_safety:	Is it aligned with ethical & safety policies?

6. task_completion: Did the assistant do what was asked?

7. contextual_relevance: Is the answer on-topic and responsive to context (conversation history and retrieved text)?

8. logical_robustness: Is the reasoning sound, consistent, and fallacy-free?

Each metric is scored from 1 (poor) to 5 (excellent).

**Why these metrics**

The is a multi-purpose assistant meaning it is designed to help with a wide range of tasks across different domains (conversational, versatile, cross-domain) judging from the nature of user questions of the dataset. For a production assistant the metric *safety policy* was necessary to ensure no harmful answers were given ensuring the assistant has the necessary guardrails. Considering it as a general purporse assistant, metrics were selected in a way that covers many use cases that the dataset contains. Many of them are fact-based user questions meaning that *grounding faithfulness* is an extremely important metric. This metric also is generally recommended as it gives us a clue about the retrieval process. *Language appropriateness* is another relevant metric for production assistants, which scores how suitable language is certain contexts. For example, when the user asks the assistant to tell a joke, the assistant shouldn't use sophisticated language.
 *Task completion* is another important metric and focuses on how helpful the assistant was to the end user and if the assistant followed users' instructions. Asking for a joke, asking for a summarization of something or a python formula indicate use cases where the user provides an instruction and upon delivering it successsfully or not is encapsulated by the above metric. *Logical robustness* is another metric that evaluates how logical or illogical answers are provided based on context. Also, it evaluates the model's tendency to proivde illogical answers in tricky questions that include false premises. *Contextual relevance* is important to ensure the model provides a relevant answer to the specific question considering the conversation history - this metric isn't designed to penalize the assistant's answers if the user decides to change the topic. It evaluates the relevance of the answers based on the whole context. *Contradiction* is another metric that evaluates if any part of the answer contradicts the context. It is generally very useful, if we want to check if the model contradicts previous answers. However, in this project, this metric did not consider conversation history.

All in all, the metrics are core and relevant for a general-purpose assistant that handles different kind of questions, ensuring that core aspects of the assistant are evaluated (safety policy, language, faithfulness). It is recommended that classification of the question types be implemented beforehand (either with an LLM or an embedding model), such as fact-based questions, conversational, safety etc. as this extra layer of grouping can inform us on the creation of more targeted and specific metrics. 
 
 **Output will be saved to**:

reports/graded_timestamp.csv

reports/report_timestamp.md

**Model Switch Guide**


Currently, the evaluation uses mistral-small-2506 via the Mistral API. To change the model:

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
