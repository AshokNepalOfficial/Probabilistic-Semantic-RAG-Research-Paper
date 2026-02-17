import matplotlib.pyplot as plt
import numpy as np
from new_final_code import ProbabilisticRecursiveRAGV4
from new_final_code import ProbabilisticRecursiveRAG
import tiktoken

def run_technical_benchmark(document, query):
    # 1. THE RESEARCH DATASET (High Noise-to-Signal Ratio)
    # 2. EXECUTE THE DISTILLATION
    # Target 60 tokens to force a ~70% reduction from the original 200+
    print("[Research] Initiating Aggressive Distillation (V4)...")
    rag_engine = ProbabilisticRecursiveRAG(target_tokens=60)
    ans, gist, logs = rag_engine.synthesize(document, query)

    # 3. METRIC EXTRACTION
    tokenizer = tiktoken.get_encoding("cl100k_base")
    initial_tokens = len(tokenizer.encode(document))
    final_tokens = len(tokenizer.encode(gist))
    
    depths = [l['depth'] for l in logs]
    tokens_at_depth = [l['tokens'] for l in logs]
    kl_gains = [l['kl'] for l in logs]
    
    # Append final state to token list for the plot
    tokens_at_depth.append(final_tokens)
    depth_labels = depths + [depths[-1] + 1]

    # ==============================================================================
    # GRAPH 1: THE RECURSIVE FUNNEL (Aggressive Token Reduction)
    # ==============================================================================
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(depth_labels, tokens_at_depth, color='#d62728', marker='o', linewidth=3, markersize=10, label='Token Count')
    plt.fill_between(depth_labels, tokens_at_depth, alpha=0.1, color='red')
    plt.axhline(y=60, color='gray', linestyle='--', label='Target Threshold')
    
    plt.title('Token Reduction: The Semantic Funnel', fontsize=14, fontweight='bold')
    plt.xlabel('Recursion Depth ($d$)', fontsize=12)
    plt.ylabel('Token Complexity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # ==============================================================================
    # GRAPH 2: INFORMATION GAIN (KL-Divergence purification)
    # ==============================================================================
    plt.subplot(1, 2, 2)
    plt.plot(depths, kl_gains, color='#1f77b4', marker='s', linewidth=3, markersize=10, label=r'Information Gain ($D_{KL}$)')
    plt.fill_between(depths, kl_gains, alpha=0.1, color='blue')
    
    plt.title('Contextual Purification: Density Gain', fontsize=14, fontweight='bold')
    plt.xlabel('Recursion Depth ($d$)', fontsize=12)
    plt.ylabel(r'KL-Divergence Score ($D_{KL}$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('images/technical_funnel.png', dpi=300)

    # ==============================================================================
    # GRAPH 3: COMPARATIVE BENCHMARK (TOKEN BURDEN)
    # ==============================================================================
    plt.figure(figsize=(8, 6))
    reduction_pct = ((initial_tokens - final_tokens) / initial_tokens) * 100
    
    labels = ['Naive RAG (Baseline)', 'Probabilistic Recursive RAG']
    values = [initial_tokens, final_tokens]
    
    bars = plt.bar(labels, values, color=['#7f7f7f', '#1f77b4'], alpha=0.8)
    plt.title('Knowledge Synthesis Efficiency', fontsize=14, fontweight='bold')
    plt.ylabel('Input Tokens Sent to LLM', fontsize=12)
    
    # Adding the specific percentage label you need for your research proof
    plt.text(1, final_tokens + 5, f"-{reduction_pct:.1f}% Tokens", 
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='#d62728')
    
    plt.savefig('images/token_reduction_benchmark.png', dpi=300)

    print("\n" + "="*60)
    print("FINAL RESEARCH GRAPHS GENERATED:")
    print(f"1. Initial Tokens: {initial_tokens}")
    print(f"2. Final Tokens:   {final_tokens}")
    print(f"3. TOKEN SAVINGS:  {reduction_pct:.2f}%")
    print("="*60)
    print("Files saved: images/technical_funnel.png, images/token_reduction_benchmark.png")


def generate_advanced_research_graphs(doc,query):
    print("[Analysis] Running Advanced Recursive Distillation...")
    rag = ProbabilisticRecursiveRAGV4(target_tokens=60)
    _, _, logs = rag.synthesize(doc, query)

    # Data Extraction
    d = [l['depth'] for l in logs]
    t = [l['tokens'] for l in logs]
    kl = [l['kl'] for l in logs]
    p = [l['avg_p'] for l in logs]
    ent = [l['avg_ent'] for l in logs]
    g_sim = [l['avg_g_sim'] for l in logs]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Parametric Technical Analysis: Probabilistic Semantic RAG', fontsize=16, fontweight='bold')

    # Curve 1: The Information Bottleneck (Tokens vs KL)
    ax1 = axs[0, 0]
    ax1.plot(d, t, 'r-o', label='Tokens (Noise)', linewidth=2)
    ax1.set_ylabel('Token Complexity', color='r')
    ax1_2 = ax1.twinx()
    ax1_2.plot(d, kl, 'b-s', label='Info Density', linewidth=2)
    ax1_2.set_ylabel('KL-Divergence (Purification)', color='b')
    ax1.set_title('1. Information Bottleneck Optimization')
    ax1.grid(True, alpha=0.3)

    # Curve 2: Entropy Decay (Noise Removal)
    ax2 = axs[0, 1]
    ax2.plot(d, ent, 'm-D', label='Shannon Entropy', linewidth=2)
    ax2.set_title('2. Semantic Entropy Decay (Noise Suppression)')
    ax2.set_ylabel('Entropy Score (H)')
    ax2.set_xlabel('Recursion Depth')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Curve 3: Mean Probabilistic Confidence
    ax3 = axs[1, 0]
    ax3.plot(d, p, 'g-^', label='Mean P(c|q)', linewidth=2)
    ax3.set_title('3. Probabilistic Confidence Gain')
    ax3.set_ylabel('Confidence Score (0-1)')
    ax3.set_xlabel('Recursion Depth')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Curve 4: Global Anchor Stability
    ax4 = axs[1, 1]
    ax4.plot(d, g_sim, 'k-p', label='Global Anchor Sim', linewidth=2)
    ax4.set_title('4. Contextual Grounding (Anchor Stability)')
    ax4.set_ylabel('Similarity to Original Corpus')
    ax4.set_xlabel('Recursion Depth')
    ax4.set_ylim(0.7, 1.0) # Show precision
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('images/advanced_research_analysis_img_2.png', dpi=300)
    print("[Success] 4-Panel Research Analysis saved as 'images/advanced_research_analysis_img_2.png'")







if __name__ == "__main__":
    # document = """
    # The James Webb Space Telescope (JWST) uses a 6.5-meter primary mirror to capture infrared light. 
    # It is currently orbiting at the second Lagrange point (L2), 1.5 million kilometers from Earth. 
    # In global economics, the price of crude oil is seeing a minor correction this week. 
    # The JWST's sunshield is the size of a tennis court and protects it from heat. 
    # A popular recipe for pasta carbonara requires guanciale, pecorino, and fresh eggs. 
    # The primary mission of the JWST is to study the first stars and galaxies formed after the Big Bang. 
    # In unrelated news, the local transit system is adding new electric buses next month. 
    # By applying recursive probabilistic RAG, we can strip these irrelevant sentences away. 
    # The history of the Eiffel Tower shows it was built for the 1889 World's Fair. 
    # The 'gist' of JWST's success is its cryogenic cooling and infrared sensitivity.
    # """
    # query = "Where is the JWST located and what is its primary mission?"



    # document = """
    # # Quantum computers use qubits to perform calculations far faster than classical computers. 
    # # One major hurdle is decoherence, where qubits lose their quantum state due to noise. 
    # # Researchers at IBM are testing new cryogenic shields to protect these circuits. 
    # # The price of coffee beans in Ethiopia has reached a five-year high this month. 
    # # In contrast, Microsoft is pursuing 'topological' qubits which are inherently stable. 
    # # Baking a sourdough bread requires a starter, flour, and a very long fermentation time. 
    # # The primary goal of quantum research is to achieve fault-tolerant computing. 
    # # In unrelated news, the local transit system is adding new high-speed rail lines. 
    # # Our recursive RAG system extracts the 'gist' by filtering these unrelated topics. 
    # # The final success of quantum scaling depends on high-density error correction.
    # # """
    # query = "What are the specific methods IBM and Microsoft use to stabilize qubits?"


    # document = """
    # Quantum computers require qubits to be kept at cryogenic temperatures. 
    # Thermal noise is the primary cause of decoherence in these systems. 
    # The price of electricity for cooling these systems is rising globally. 
    # Researchers at MIT are developing new superconducting materials that operate at higher temperatures. 
    # NASA is also interested in these materials for satellite communication. 
    # In unrelated news, the local transit system announced new schedules for Monday. 
    # The MIT team found that by layering these materials, they can achieve better qubit stability. 
    # This layering method reduces the context of external noise interference by 40%. 
    # Agriculture in the Midwest is currently facing a minor drought. 
    # Synthesis of these specific qubit stability methods is the key to scaling quantum power.
    # """
    
    # query = "What specific methods are MIT researchers using to improve qubit stability?"



    # document = """
    # Large-scale synthesis is often blocked by context window limits. 
    # Quantum computing researchers are focusing on topological insulators. 
    # The weather in the Atlantic is showing signs of a heavy hurricane season. 
    # By using recursive probabilistic chunking, we can discard 80% of noise. 
    # Topological insulators act as conductors on the surface but insulators inside. 
    # In local news, a new library opened downtown last Tuesday. 
    # This duality allows for qubits that are protected from local environmental noise. 
    # Stabilizing these qubits is the primary barrier to building a 1-million qubit system. 
    # High-density gist extraction ensures the LLM receives only 'Knowledge Atoms'.
    # """
    
    # query = "What is the role of topological insulators in qubit stability?"
    



    # =========================
    # Context Document
    # =========================

    document = """
    This research proposes a Probabilistic Semantic Recursive Distillation framework
    for large-scale knowledge synthesis. The main challenge addressed is the high
    Noise-to-Signal Ratio (NSR) in Retrieval-Augmented Generation (RAG) systems.

    Traditional RAG pipelines rely on fixed-length chunking and single-pass similarity
    retrieval, which often results in semantic fragmentation and inclusion of irrelevant
    contextual padding. To overcome this limitation, the proposed method treats retrieval
    as a multi-stage information distillation process guided by information theory.

    The methodology consists of four main phases. Phase I introduces Adaptive Semantic
    Boundary Partitioning, where semantic boundaries are detected dynamically using
    embedding-based cosine distance between adjacent sentences. This ensures that text
    chunks preserve conceptual coherence rather than arbitrary length.

    Phase II applies a Tri-Factor Probabilistic Weighting scheme. Each chunk is scored
    using a unified probability function that combines local semantic relevance to the
    query, entropy-based uncertainty filtering using Shannon entropy, and global semantic
    consistency via a corpus-level anchor embedding.

    Chunks that exceed a relevance threshold are aggregated into a Combined Paragraph,
    which serves as an intermediate condensed representation of the source material.

    Phase III introduces Recursive Gist Distillation, where the combined paragraph is
    treated as a new document and reprocessed using finer-grained chunking. Information
    gain at each recursion depth is validated using Kullback–Leibler (KL) divergence,
    providing quantitative evidence of semantic purification.

    The recursion terminates automatically based on token budget constraints or semantic
    saturation, ensuring stability and preventing over-compression.

    In Phase IV, the final distilled semantic atoms are provided to a local large language
    model for synthesis. Because most noise has already been removed, the model focuses
    on reasoning and synthesis rather than filtering irrelevant information.

    Overall, the framework operationalizes the Information Bottleneck Principle for RAG
    systems and provides a mathematically grounded solution to scalable knowledge
    retrieval and synthesis.
    """

    # =========================
    # Query (Research Question)
    # =========================

    query = """
    How does the proposed Probabilistic Semantic Recursive Distillation framework
    reduce noise and improve knowledge density in retrieval-augmented generation systems?
    """
    run_technical_benchmark(document, query)
    generate_advanced_research_graphs(document,query)

