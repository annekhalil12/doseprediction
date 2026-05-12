# Thesis Writing Guidelines

## Format
- ACM double-column format, Overleaf template
- Maximum 10 pages (references start top of second column of page 10, appendix excluded)
- ACM reference format

---

## Section Requirements

### 1. Abstract (50–200 words, no citations)
Five-element structure:
1. General statement introducing the research area
2. Specific problem / research gap
3. Review of existing models/theory related to the problem
4. Outline of proposed approach
5. Summary of evaluation results and scientific contribution

Include keywords below the abstract.

### 2. Introduction
- Position topic in scientific field (task-oriented or subject-oriented)
- Problem statement: why is it scientifically relevant? Make trade-offs explicit
- Mention key papers and explicitly formulate the research gap
- Research question tree (depth 3, branching factor 3, numbered):
  - Root RQ: general, not yes/no, preferably "to what extent", comparative vs. baseline
  - Sub-RQs (SRQ1–3): provide evidence to answer root RQ
  - Leaf sub-RQs: concrete enough that readers can predict the shape of the answer (e.g. a table of MAE scores)
- Every paragraph/section must connect to a (S)RQ — and every (S)RQ must be answered

### 3. Related Work
Organise into 3–4 subsections, each linked to a theme/RQ group:
- First subsection: state of the art on main RQ (~5–10 references, 2–3 sentences each)
- Other subsections: technical/methodological themes
- Must cover: techniques used by others, results, evaluation metrics, datasets, terminology
- Show SOTA awareness; justify that you are using the best available methods

Citation key format: `nnnn:aaaayy` (first 4 chars of first author + first 4 chars of first non-stop title word + 2-digit year)

### 4a. Methodology
- Describe what you do AND justify why (over alternatives)
- Not a textbook section — focus on your specific addition/change
- Address reliability, validity, and research design credibility
- Can be brief if there is no novel methodological contribution

### 4b. Experimental Setup
Two functions: (1) allow replication, (2) give a clear picture of datasets.

Must include:
- Data preprocessing steps
- All hyperparameter settings
- Train/validation/test split strategy
- Exact metric calculation details (e.g. macro vs. micro, per-structure vs. aggregate)
- EDA graphs + basic dataset statistics

Use appendix or notebook reference if more detail is needed.

### 5. Results
- Structure: one table/figure per research question + elaborate caption
- Captions must be self-contained — reader should understand results from question + figure alone
- Keep prose minimal; numbers must be precise (no "very", "small", "large")
- Do not answer RQs here — that goes in Discussion/Conclusion

### 6. Discussion
Subsections:
1. Compare results with prior work (concrete numbers, references, significance/confidence intervals)
2. Explain unexpected results — most likely reasons, reference comparable studies
3. Limitations framed as future research directions (use: reproducibility, scalability, generalisability, reliability, validity)
4. Alternative conclusions consistent with results; bridge to conclusion

### 7. Conclusion (max half a page)
Structure:
1. Recap scientific relevance and problem statement
2. Answer each research question (rephrase indirectly: "This research aimed to…")
3. Qualify conclusions with limitations
4. State value vs. state of the art — what does this add?
5. Future work: argue most promising direction (don't list what "must" be done)
