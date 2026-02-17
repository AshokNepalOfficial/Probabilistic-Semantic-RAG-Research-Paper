
import numpy as np
from scipy.stats import entropy
from llama_index.core.node_parser import SentenceSplitter,SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode

# ==================================================================================
# RESEARCH PROJECT: Probabilistic Semantic Chunked RAG for Large Scale Synthesis
# Aggressive Distillation with Information Density Check
# ==================================================================================

class ProbabilisticRecursiveRAG:
    def __init__(self, model_name="gpt-oss:20b", target_tokens=100):
        self.model_name = model_name
        self.target_tokens = target_tokens
        
        print("[System] Loading Scientific Encoders...")
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model=model_name, request_timeout=180.0)
        
        # Multi-Granular Parsers
        self.semantic_parser = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
        )
        self.fine_splitter = SentenceSplitter(chunk_size=35, chunk_overlap=0)
        
        self.stats_log = []

    def _get_entropy(self, emb):
        """Shannon Entropy for noise detection."""
        abs_emb = np.abs(emb)
        prob_dist = abs_emb / np.sum(abs_emb)
        return entropy(prob_dist)

    def _calculate_kl_div(self, p_emb, q_emb):
        """KL-Divergence for Information Gain proof."""
        p = np.abs(p_emb) / np.sum(np.abs(p_emb))
        q = np.abs(q_emb) / np.sum(np.abs(q_emb))
        return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

    def _prob_score(self, text, query, anchor_emb, depth):
        """Unified Probabilistic Metric P(c|q)."""
        c_emb = np.array(self.embed_model.get_text_embedding(text))
        q_emb = np.array(self.embed_model.get_query_embedding(query))
        
        l_sim = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
        ent = self._get_entropy(c_emb)
        certainty = 1.0 / (1.0 + ent)
        g_sim = np.dot(anchor_emb, c_emb) / (np.linalg.norm(anchor_emb) * np.linalg.norm(c_emb))
        
        # Adaptive Weights
        alpha = 0.5 + (depth * 0.1)
        beta = 0.3 - (depth * 0.05)
        gamma = 0.2 - (depth * 0.05)
        
        return (alpha * l_sim) + (beta * certainty) + (gamma * g_sim), c_emb

    def recursive_funnel(self, context, query, anchor_emb, depth=0):
        """
        RECURSIVE FUNNEL (FIXED):
        Prioritizes Token Reduction (Target Tokens) over early semantic exit.
        """
        tokens = int(len(context.split()) * 1.3)
        
        # Mandatory exit conditions
        if (tokens <= self.target_tokens and depth >= 1) or depth > 5:
            print(f"[Depth {depth}] DISTILLATION COMPLETE: Gist size {tokens} tokens.")
            return context

        print(f"[Depth {depth}] Reducing context... Current Tokens: {tokens}")

        # 1. GRANULARITY INVERSION (Semantic -> Propositional)
        if depth == 0:
            nodes = self.semantic_parser.get_nodes_from_documents([TextNode(text=context)])
            chunks = [n.get_content() for n in nodes]
        else:
            chunks = self.fine_splitter.split_text(context)

        # 2. PROBABILISTIC FILTERING (AGGRESSIVE)
        scored = []
        for c in chunks:
            p, c_emb = self._prob_score(c, query, anchor_emb, depth)
            # Higher threshold for deeper levels to kill more noise
            threshold = 0.45 + (depth * 0.02)
            if p > threshold:
                scored.append((c, p, c_emb))
        
        if not scored: return context
        
        # 3. SELECTION (The Gist Extraction)
        scored.sort(key=lambda x: x[1], reverse=True)
        # Aggressive limit from V2: reduces atoms as we go deeper
        top_n = max(2, 6 - depth)
        best_chunks = [s[0] for s in scored[:top_n]]
        new_context = ". ".join(best_chunks)

        # LOGGING METRICS (For Technical Analysis)
        p_emb = self.embed_model.get_text_embedding(context)
        q_emb = self.embed_model.get_text_embedding(new_context)
        kl = self._calculate_kl_div(q_emb, p_emb)
        self.stats_log.append({"depth": depth, "tokens": tokens, "kl": kl})

        return self.recursive_funnel(new_context, query, anchor_emb, depth + 1)

    def synthesize(self, document, query):
        print("\n--- INITIATING RECURSIVE SYNTHESIS ---")
        anchor_emb = np.array(self.embed_model.get_text_embedding(document))
        final_gist = self.recursive_funnel(document, query, anchor_emb)
        
        prompt = f"Using this distilled context, synthesize a concise answer:\n{final_gist}\n\nQuery: {query}"
        response = self.llm.complete(prompt)
        return str(response), final_gist, self.stats_log

class ProbabilisticRecursiveRAGV4:
    def __init__(self, model_name="gpt-oss:20b", target_tokens=80):
        self.model_name = model_name
        self.target_tokens = target_tokens
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.llm = Ollama(model=model_name, request_timeout=180.0)
        self.semantic_parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model)
        self.fine_splitter = SentenceSplitter(chunk_size=35, chunk_overlap=0)
        self.stats_log = []

    def _get_entropy(self, emb):
        abs_emb = np.abs(emb)
        return entropy(abs_emb / np.sum(abs_emb))

    def _calculate_kl_div(self, p_emb, q_emb):
        p = np.abs(p_emb) / np.sum(np.abs(p_emb))
        q = np.abs(q_emb) / np.sum(np.abs(q_emb))
        return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

    def _prob_score(self, text, query, anchor_emb, depth):
        c_emb = np.array(self.embed_model.get_text_embedding(text))
        q_emb = np.array(self.embed_model.get_query_embedding(query))
        l_sim = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
        ent = self._get_entropy(c_emb)
        g_sim = np.dot(anchor_emb, c_emb) / (np.linalg.norm(anchor_emb) * np.linalg.norm(c_emb))
        
        score = (0.5 * l_sim) + (0.3 * (1.0 / (1.0 + ent))) + (0.2 * g_sim)
        return score, c_emb, ent, g_sim

    def recursive_funnel(self, context, query, anchor_emb, depth=0):
        tokens = int(len(context.split()) * 1.3)
        if (tokens <= self.target_tokens and depth >= 1) or depth > 5:
            return context

        chunks = self.semantic_parser.get_nodes_from_documents([TextNode(text=context)]) if depth == 0 else [TextNode(text=c) for c in self.fine_splitter.split_text(context)]
        
        scored = []
        for n in chunks:
            p, c_emb, ent_val, g_sim = self._prob_score(n.get_content(), query, anchor_emb, depth)
            if p > (0.44 + (depth * 0.02)):
                scored.append({'text': n.get_content(), 'p': p, 'ent': ent_val, 'g_sim': g_sim})
        
        if not scored: return context
        scored.sort(key=lambda x: x['p'], reverse=True)
        top_atoms = scored[:max(2, 6 - depth)]
        new_context = ". ".join([s['text'] for s in top_atoms])

        # NEW: Logging expanded metrics for advanced curves
        p_emb = self.embed_model.get_text_embedding(context)
        q_emb = self.embed_model.get_text_embedding(new_context)
        
        self.stats_log.append({
            "depth": depth,
            "tokens": tokens,
            "kl": self._calculate_kl_div(q_emb, p_emb),
            "avg_p": np.mean([s['p'] for s in top_atoms]),
            "avg_ent": np.mean([s['ent'] for s in top_atoms]),
            "avg_g_sim": np.mean([s['g_sim'] for s in top_atoms])
        })

        return self.recursive_funnel(new_context, query, anchor_emb, depth + 1)

    def synthesize(self, document, query):
        anchor_emb = np.array(self.embed_model.get_text_embedding(document))
        final_gist, logs = self.recursive_funnel(document, query, anchor_emb), self.stats_log
        prompt = f"Distilled Context:\n{final_gist}\n\nQuery: {query}"
        return str(self.llm.complete(prompt)), final_gist, logs