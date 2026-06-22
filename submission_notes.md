# Submission Notes: Property Listing Classifier

## 1. Assumptions Made
- **Acquisitions Context vs. Hard Rules**: I reviewed Harkalm's acquisitions criteria and used them as contextual guidelines rather than hard, keyword-matching filters. I tried to focus on what the property's *actual* or *intended* use is, rather than just matching text patterns.
- **Intended Use over Adjacencies**: I assumed that just because a property is near a school or has an SEN school listed as a "point of interest" in the agent data, that doesn't make the property itself an SEN school. The listing must describe a property actively set up for or let to an education provider.

## 2. Handling Ambiguity
- **No Forced Classifications**: Real estate agent copy is notoriously messy and speculative (often claiming a site is "suitable for a variety of alternative uses like daycare, clinics, or gyms, subject to planning"). I instructed the model not to force these into active classifications if there's no actual history of that use. In these cases, it defaults to `None` or flags the confidence as `Low` or `Medium`.
- **Vacant vs. Operational**: If a property has the right planning consent (like F1/D1) but is completely vacant and needs a full refit, it’s flagged as a conversion opportunity rather than a confident match. I kept the confidence lower here so the acquisitions team knows it's a project rather than a ready-to-go acquisition.

## 3. What I'd Improve with More Time
- **Batching & Async Calls**: Right now, the script processes listings sequentially. I'd implement async requests using `asyncio` to speed up processing for larger datasets.
- **Few-Shot Examples**: Adding a few hand-labeled examples of edge cases directly into the system prompt would help align the model's reasoning on highly ambiguous listings.
- **Evaluation Dataset**: I'd set up a small, hand-labeled "golden dataset" of 50-100 listings to run automated tests and calculate accuracy, precision, and recall before deploying prompt updates.
- **Embeddings & RAG**: Instead of hardcoding prompt context, I would generate embeddings of our past successful acquisitions and retrieve similar properties dynamically to feed to the LLM as reference.
- **Confidence Calibration**: I'd look into pulling logprobs (token probabilities) from the OpenAI API to mathematically calibrate the `High/Medium/Low` confidence scores.
- **Cost & Latency Optimization**: Agent descriptions can be massive walls of text. Trimming down the input tokens to only relevant sections and testing smaller, cheaper models (like Llama 3 or GPT-3.5) would save money and run faster.
