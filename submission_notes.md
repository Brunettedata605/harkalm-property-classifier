# Submission Notes: Property Listing Classifier

## 1. Assumptions Made
- **Contextual Guidance vs. Rigid Rules**: Harkalm's three core operating sectors (nurseries, SEN schools, and food stores) were used as contextual guides to evaluate a property's fit, rather than rigid keyword matching rules.
- **Priority of Intended Use**: Property classification prioritized the actual, historical, or primary intended use. We assumed that nearby features (e.g., local primary schools listed as points of interest) or adjacent tenants do not determine the subject property's classification unless it is actively let to/operated by a target user.

## 2. Handling of Ambiguous Listings
- **Avoiding Forced Classifications**: Agent copy frequently hedges by listing speculative alternative uses ("suitable for nursery, clinic, or gym, subject to planning"). We deliberately avoided forcing these into active categories, classifying them as `None` or utilizing `Low` or `Medium` confidence levels to reflect the planning/market uncertainty.
- **Vacancy & Planning Consent**: Properties with relevant use classes (e.g., D1/F1) but no established operations, fit-out, or active operators were flagged as `None` with `Low` confidence, ensuring acquisitions teams can distinguish between active deals and raw conversion opportunities.

## 3. Future Improvements
With more time, the following enhancements would be made:
- **Batching & Async Requests**: Implement concurrent request processing to scale classification throughput.
- **Few-Shot Prompting**: Introduce curated, high-quality examples (few-shot learning) in the prompt to align classification reasoning on edge cases.
- **Evaluation Dataset**: Build a small, golden test set of labeled listings to calculate classification precision, recall, and accuracy metrics.
- **Embeddings & Semantic Search (RAG)**: Use embeddings to dynamically retrieve the most similar historical deals from Harkalm's database as reference context for the LLM.
- **Confidence Calibration**: Apply temperature scaling or logprob analysis to calibrate confidence scores.
- **Cost & Latency Optimization**: Optimize token usage by trimming verbose agent copy and benchmarking smaller open-source models (e.g., Llama 3) for lower operational costs.
