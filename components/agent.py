def simple_extract(texts):
    nodes = set()
    edges = []

    for text in texts:
        words = text.split()
        for i in range(len(words)-1):
            a = words[i].strip(",.")
            b = words[i+1].strip(",.")
            if a.istitle() and b.istitle():
                nodes.add(a)
                nodes.add(b)
                edges.append({"source": a, "target": b, "relation": "related"})
    return list(nodes), edges

def extract_topic_label(texts, embed_model):
    # Simple heuristic: pick top keywords from all texts in cluster
    from collections import Counter
    import re
    words = re.findall(r"\b\w+\b", " ".join(texts).lower())
    stopwords = set(["the", "and", "of", "to", "in", "a", "is", "for", "on", "with", "as", "by", "an", "be", "are", "at", "from", "that", "this", "it", "or", "was", "which", "but", "has", "have", "not", "can", "will", "if", "their", "more", "also", "other", "than", "such", "may", "used", "use", "using", "into", "these", "its", "been", "were", "we", "they", "one", "all", "each", "most", "some", "any", "our", "out", "so", "do", "does", "did", "had", "should", "would", "could", "about", "after", "before", "between", "during", "over", "under", "up", "down", "off", "then", "there", "when", "where", "who", "what", "why", "how"])
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    most_common = Counter(keywords).most_common(3)
    label = ", ".join([w for w, _ in most_common])
    return label.title() if label else "Topic"

def run_agent(query, store, embed_model, steps=2, all_chunks=None, num_clusters=None, cluster_chunks_map=None):
    # If all_chunks is provided and query is None, cluster and extract from all_chunks
    if all_chunks is not None and query is None:
        from sklearn.cluster import KMeans
        import numpy as np

        embeddings = embed_model.encode(all_chunks)
        n_clusters = num_clusters if num_clusters is not None else (min(8, len(all_chunks)) if len(all_chunks) > 2 else 1)
        n_clusters = min(n_clusters, len(all_chunks)) if len(all_chunks) > 0 else 1
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            labels = kmeans.fit_predict(embeddings)
            clusters = {i: [] for i in range(n_clusters)}
            for idx, label in enumerate(labels):
                clusters[label].append(all_chunks[idx])
            # Generate topic labels for each cluster
            cluster_labels = []
            cluster_chunks_map = {}
            for i in range(n_clusters):
                label = extract_topic_label(clusters[i], embed_model)
                cluster_labels.append(label)
                cluster_chunks_map[label] = clusters[i]
            nodes = cluster_labels
            edges = []
            centroids = kmeans.cluster_centers_
            for i in range(n_clusters):
                for j in range(i+1, n_clusters):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    if dist < 0.7:
                        edges.append({"source": cluster_labels[i], "target": cluster_labels[j], "relation": "similar"})
            return nodes, edges, cluster_chunks_map
        else:
            label = extract_topic_label(all_chunks, embed_model)
            return [label], [], {label: all_chunks}

    # If expanding a cluster node, use only that cluster's chunks
    if all_chunks is not None and query is not None and cluster_chunks_map is not None:
        # query is the cluster label
        cluster_texts = cluster_chunks_map.get(query, [])
        # Use simple_extract or agentic logic on cluster_texts
        from collections import Counter
        from components.agent import simple_extract
        nodes, edges = simple_extract(cluster_texts)
        return nodes, edges, None

    # Default: original agentic search
    current_query = query
    all_nodes = set()
    all_edges = []

    for _ in range(steps):
        q_emb = embed_model.encode(current_query)
        retrieved = store.search(q_emb, k=5)

        # Import simple_extract from app if not already imported
        nodes, edges = simple_extract(retrieved)

        all_nodes.update(nodes)
        all_edges.extend(edges)

        if nodes:
            # pick next node (longest = more informative)
            current_query = max(nodes, key=len)

    return list(all_nodes), all_edges