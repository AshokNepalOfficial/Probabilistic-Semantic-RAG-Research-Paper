# 3. Methodology

## 3.1 Theoretical Framework: The Information Bottleneck Principle
The fundamental challenge in Large-Scale Knowledge Synthesis is **Semantic Dilution** caused by high **Noise-to-Signal Ratios (NSR)** in retrieved context. We model our architecture based on the **Information Bottleneck (IB) Principle** (Tishby et al.), which posits that an optimal representation $S$ (the "Gist") must maximize mutual information with the query $q$ while minimizing the description length (token complexity) of the source $D$. 

Formally, we define our objective function as a constrained optimization problem:
$$\mathcal{L}(S, q) = \max_{X \subseteq \mathcal{D}} [I(X; q) - \beta \cdot |X|]$$
where $I(X; q)$ denotes mutual information, $|X|$ represents token complexity, and $\beta$ is a Lagrange multiplier controlling the trade-off between compression and relevance.

---

## 3.2 Phase I: Adaptive Semantic Partitioning (Step 1)
To mitigate **Semantic Fragmentation** (the loss of conceptual integrity caused by fixed-window chunking), we implement **Adaptive Boundary Partitioning**.

Let document $D$ be represented as a discrete sequence of sentence embeddings $\mathbf{V} = \{v_1, v_2, \dots, v_n\}$ in a $d$-dimensional latent space $\mathbb{R}^d$. We quantify the semantic transition $\delta$ between adjacent vectors using a cosine-distance-based shift metric:
$$\delta_i = 1 - \frac{v_i \cdot v_{i+1}}{\|v_i\| \|v_{i+1}\|}$$
Boundaries are established where $\delta_i$ exceeds a percentile-based threshold $\tau$. This ensures the initial chunks $C = \{c_1, \dots, c_m\}$ represent conceptually independent thematic units, preserving the logical flow of the original knowledge base.

---

## 3.3 Phase II: Tri-Factor Probabilistic Weighting (Steps 2–4)
Traditional RAG architectures rely exclusively on cosine similarity, which fails to differentiate between "vague" and "dense" knowledge. We propose a **Unified Probabilistic Score $P(c_i|q)$** to quantify information precision.

### 3.3.1 The Probabilistic Scoring Formula
For each candidate chunk $c_i$, we compute a score based on three distinct latent dimensions:
$$P(c_i | q) = \omega_1 \cdot \mathcal{S}_{local}(c_i, q) + \omega_2 \cdot (1 - \hat{\mathcal{H}}(c_i)) + \omega_3 \cdot \mathcal{S}_{global}(c_i, \mathcal{G})$$
Subject to the constraint $\sum \omega_j = 1$.

1.  **Local Relevance ($\mathcal{S}_{local}$):** Measures the direct semantic affinity between the chunk and the query.
2.  **Epistemic Certainty ($\hat{\mathcal{H}}$):** We utilize **Shannon Entropy** to measure semantic noise. Focused, fact-dense chunks exhibit low entropy, while "rambling" passages exhibit high entropy:
    $$\mathcal{H}(c_i) = - \sum_{j=1}^{d} p_j \log p_j, \quad p_j = \frac{|v_j|}{\sum_{k=1}^{d} |v_k|}$$
3.  **Global Anchoring ($\mathcal{S}_{global}$):** Chunks are compared against $\mathcal{G}$ (the embedding of the entire source document). This acts as a **Semantic Compass**, preventing **Semantic Drift** during recursion.

Chunks satisfying $P(c_i | q) > \lambda$ are selected and aggregated into a **Combined Paragraph (CP)**.

---

## 3.4 Phase III: Recursive Information Distillation (Steps 5–7)
The core innovation of this research is the **Recursive Semantic Funnel**. We propose that "Remembering" large-scale knowledge requires iterative refinement to transition from **Thematic Chunks** to **Propositional Atoms**.

### 3.4.1 Granularity Inversion
The system treats the **CP** as a new source and recursively re-chunks it. As the recursion depth $d$ increases, the granularity of the parser shifts:
*   **Depth $0$:** Semantic Topic Splitting (Coarse Filtering).
*   **Depth $d > 0$:** Propositional Atomization (Fine-grained Sentence Extraction).

### 3.4.2 Mathematical Validation: KL-Divergence
To quantify the "Purification" of the context, we calculate the **Kullback-Leibler (KL) Divergence** between the parent context distribution $P$ and the distilled gist distribution $Q$:
$$D_{KL}(Q \| P) = \sum_{x \in \mathcal{X}} Q(x) \log \left( \frac{Q(x)}{P(x) + \epsilon} \right)$$
An increase in $D_{KL}$ across depths provides empirical evidence of **Contextual Purification**—the mathematical concentration of semantic mass around the relevant "Knowledge Atoms."

### 3.4.3 Semantic Convergence and Early Exit
Distillation terminates autonomously when **Information Density** saturates. This is measured by the cosine similarity between successive iterations. If $\cos(S_d, S_{d-1}) > (1 - \epsilon)$, the signal is deemed pure, and the recursion exits.

---

## 3.5 Phase IV: High-Density Synthesis (Steps 8–9)
The final distilled atoms represent the **Maximum Information Density (MID)** of the original corpus.

### 3.5.1 LLM-Based Knowledge Synthesis
The purified gist is injected into a local **gpt-oss:20b** model via Ollama. By presenting the LLM with a context window consisting of near-zero noise ($80\%+$ token reduction in large datasets), we transform the model’s operation from a **Filter** to a **Synthesizer**. The model focuses on the cross-logical connections between "Gist Atoms" to generate the **Final Synthesized Knowledge Output $S$**.

---

## 3.6 Outstanding Technical Contributions
1.  **Recursive Propositional Distillation:** An iterative loop that auto-optimizes the depth of information purification based on a target token budget.
2.  **Entropy-Aware Noise Suppression:** Utilizing Shannon Entropy to mathematically penalize and excise diffuse semantic content.
3.  **Global Anchor Grounding:** Implementing a semantic compass using corpus-level embeddings to prevent drift during large-scale synthesis.
4.  **Information Density Tracking:** Providing a framework to visualize the "Purification" of knowledge via KL-Divergence Gain, ensuring the context window is populated with **Maximum Information Density (MID)**.