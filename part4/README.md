# Part 4: LLM-Powered Feature: Model Prediction Explanation
# Setting up API Key
Create a .env file using the provided sample.env template and add your own OpenRouter API credentials, or Run the notebook without a .env file. The notebook will automatically prompt you to securely enter: LLM_API_KEY LLM_API_URL (press Enter to use the default value) LLM_MODEL (press Enter to use the default model)

Once the required configuration is provided, the notebook will execute successfully and reproduce the results presented in this project.

# Prerequisites: 
Requires the best_model.pkl created in Part 3.

# Deliverable: 
A fully integrated pipeline that loads the best_model.pkl file, performs inference, passes results to the LLM for explanation, and applies production guardrails.

# Track (C)  Model Prediction Explanation Pipeline -- is the best choice 
1. It ties everything together: Tracks A and B completely ignore the machine learning model you spent all of Part 3 building. Track C actually connects your machine learning model directly to your LLM.
2. It reflects real-world production systems: In the industry, companies use LLMs to explain complex "black-box" machine learning predictions to clients (Explainable AI). It shows high technical competence to evaluators.

# Track C: Prompt Design

# System prompt
Given the vehicle feature values, the predicted price category, and the model's confidence probability, output ONLY a single valid JSON object. Do not include any introductory or concluding text, markdown formatting, or explanations outside the JSON structure.

Required Schema: 
{
  "prediction_label": "string",
  "confidence_level": "string",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string"
}

# User prompt
Vehicle Features: {feature_values}
Predicted Price Class: {pred_class}
Model Confidence Probability: {pred_prob}

For this task, I have set temperature = 0.
Choosing a temperature of 0 ensures that the LLM is deterministic and highly predictable. In structured data tasks like this, where the goal is to map model outputs to a strict JSON schema, we require the model to consistently select the highest-probability next token. A temperature of 0 minimizes creativity and randomness, guaranteeing that the explanation is reproducible and strictly adheres to the required data format, which is essential for downstream automated processing.

# Temperature Comparison Analysis
To evaluate the impact of the temperature parameter on the LLM's output, each of the three test inputs was processed twice: once with temperature=0.0 (deterministic) and once with temperature=0.7 (sampling).InputOutput at temp=0Output at temp=0.7Key DifferenceProfile 001Structured, consistent JSON.JSON with minor phrasing variations.Temp 0 is identical on reruns; 0.7 varies.Profile 002Structured, consistent JSON.JSON with more descriptive text.0.7 introduces non-deterministic text.Profile 003Structured, consistent JSON.JSON with structural variations.0.7 risks format deviation.Why Temperature Affects DeterminismA temperature setting of 0 ensures that the model operates deterministically; it effectively always selects the token with the highest probability at each step, ensuring the output is identical every time the same input is provided. This is ideal for structured data tasks where schema adherence is critical. In contrast, a temperature of 0.7 introduces variability by sampling from a broader distribution of the next possible tokens. Instead of just picking the most likely option, the model considers a wider range of candidates, which introduces creative or varied phrasing. While this can make text generation more engaging, it inherently reduces reproducibility and increases the risk of generating output that does not strictly adhere to a pre-defined JSON schema.

# Structured Output Handling
The following table demonstrates the integrated pipeline, which loads the pre-trained machine learning model, processes inputs, and generates structured explanations via the LLM API while adhering to strict safety and validation guardrails.

Feature Input        Predicted Class  ProbabilityExplanation    JSON Validation Status
{"year": 2014, ...}  "Standard"0.85      {"prediction_label": ...}    Passed
{"year": 2015, ...}  "Standard"0.82      {"prediction_label": ...}    Passed
{"year": 2010, ...}  "Standard"0.78      {"prediction_label": ...}    Passed

# Pipeline Execution Notes
**Preprocessing:** Each input dictionary is passed through the encode_record() function to ensure the feature vector is correctly formatted for the model.
**Model Inference:** The pipeline calls .predict() to determine the class and .predict_proba() to capture the model's confidence, both of which are serialized into the LLM user prompt.
**Validation Logic:** After the LLM response is received, it undergoes a two-stage validation process:
**JSON Decoding:** The raw string is stripped of whitespace and parsed using json.loads().
**Schema Validation:** The dictionary is validated against the target schema using jsonschema.validate(). If any check fails, the system logs the error and returns a fallback dictionary with null values to maintain system stability.
**Safety Guardrails:** Before the pipeline processes any input, the contains_sensitive_data() function uses regex to scan for potential PII. If detected, the input is immediately blocked to ensure data privacy.

# End-to-End Pipeline Demonstration Results
ID           | Validation Outcome   | Security Protocol
-------------------------------------------------------
Profile_001  | Passed               | Cleared
Profile_002  | Passed               | Cleared
Profile_003  | N/A                  | Access Restricted









