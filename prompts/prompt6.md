Prompt 6: Clustering architecture
Act as a senior machine-learning engineer.

Design the deterministic topic-clustering pipeline for WikiPulse.

Do not use the LLM to calculate article similarity or traffic metrics.

Requirements:

1. Construct one document per article from:
- normalized title
- optional Wikipedia summary

2. Extract TF-IDF unigrams and bigrams.

3. Apply a ranking-time title boost without duplicating title text.

4. Prevent artificial bigrams across the title/summary boundary.

5. Generate local sentence embeddings using MiniLM.

6. Cluster articles using deterministic agglomerative clustering.

7. Compute:
- total pageviews
- mean pairwise similarity
- minimum pairwise similarity
- cohesion score
- summary coverage

8. Produce stable cluster IDs based on ordered article membership.

9. Preserve article and preparation ordering.

10. Add tests for:
- empty input
- one article
- repeated titles
- missing summaries
- deterministic labels
- embedding-shape validation
- stable cluster IDs
- threshold boundaries
- no source-object mutation

First propose the exact algorithm and thresholds.
Do not edit files until the plan is approved.




