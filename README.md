# Vector Databases, Embeddings, and ANN Indexes — Personal Notes (README)

> **Purpose:** My own notes — what I understood so far about vector databases, embeddings, chunking, and approximate nearest neighbor (ANN) indexing.

---

## Table of Contents

- [What is a Vector Database?](#what-is-a-vector-database)
- [Similarity Search (Cosine, Dot, L2)](#similarity-search-cosine-dot-l2)
- [Embeddings](#embeddings)
  - [Dense vs Sparse](#dense-vs-sparse)
  - [What the Output Vector Represents](#what-the-output-vector-represents)
- [When Data Is Large: Chunking](#when-data-is-large-chunking)
  - [Chunking Strategies](#chunking-strategies)
  - [Why Chunking Matters](#why-chunking-matters)
- [What if the Data Is Not Text?](#what-if-the-data-is-not-text)
- [How Vector Search Is Made Fast: Indexes](#how-vector-search-is-made-fast-indexes)
  - [HNSW (Hierarchical Navigable Small World)](#hnsw-hierarchical-navigable-small-world)
  - [DISKANN (Vamana Graph + Product Quantization)](#diskann-vamana-graph--product-quantization)
- [Practical Patterns](#practical-patterns)
- [Common Pitfalls](#common-pitfalls)
- [Glossary](#glossary)

---

## What is a Vector Database?

A **vector database** stores items as **vectors (embeddings)** and supports **nearest-neighbor search**: given a query vector, it finds stored vectors that are “closest” under a similarity/distance metric.

Key idea:

- In a “classic” DB query, you typically search by **exact matches** or strict conditions (`id = ...`, `status = ...`, `price < ...`).
- In a vector DB, you search by **similarity**: items don’t need to match exactly; they just need to be *close enough* in vector space.

In practice, vector databases often also support:

- **Metadata filtering** (e.g., `tenant_id`, `type`, `created_at`)
- **Hybrid search** (combining vector similarity + keyword/BM25)

---

## Similarity Search (Cosine, Dot, L2)

The most common similarity approaches:

- **Cosine similarity** — compares direction (angle) between vectors.
- **Dot product** — similar to cosine if vectors are normalized; also used when magnitude encodes something meaningful.
- **Euclidean (L2) distance** — compares absolute distance in vector space.

### Cosine similarity

Given vectors `a` and `b`:

\[
\cos(\theta) = \frac{a \cdot b}{\|a\|\|b\|}
\]

Interpretation:

- `cos ≈ 1` → vectors point in the **same direction** (high similarity)
- `cos ≈ 0` → vectors are **orthogonal** (unrelated)
- `cos ≈ -1` → vectors point in **opposite directions**

Why “direction” matters:

- If you scale a vector up/down (make it longer/shorter), the **cosine similarity stays the same** (because scaling cancels out in the normalization).
- Many embedding pipelines **normalize vectors** (unit length), which makes cosine and dot product effectively equivalent.

---

## Embeddings

An **embedding** is a function/algorithm that transforms data into a vector:

- Text → vector
- Image → vector
- Audio → vector
- Code → vector
- etc.

The goal: in vector space, **semantic/functional similarity** tends to correspond to **geometric closeness**.

### Dense vs Sparse

1. **Dense embeddings**
   - Most elements are non-zero.
   - Typical dimensionalities: `256`, `512`, `768`, `1536`, `3072`, etc.
   - Very common for semantic search and neural retrieval.

2. **Sparse embeddings**
   - Many elements are `0`.
   - Very high dimensionality (often `10k` → `1,000k+`).
   - Historically common in classical search (e.g., TF-IDF/BM25), and also in modern “sparse neural” approaches.
   - Useful when exact terms/rare keywords matter a lot.

> Dense ≈ meaning/semantics  
> Sparse ≈ lexical matching / exact tokens  
> Hybrid often gives the best of both.

### What the Output Vector Represents

Example:

```python
v = [0.13, ..., 0.875]  # dimensionality = n
```

 ## Building an Index in a Vector Database

Vector search becomes slow if we do it “naively”:

- compute distance(query_vector, **every** vector in the DB)
- sort
- return top-K

That’s **O(N)** per query, which doesn’t scale.  
So vector databases build **ANN indexes** (Approximate Nearest Neighbors):

- **Approximate** = not always the mathematically perfect nearest vectors
- but usually *very high recall*
- and *orders of magnitude faster* in practice

---

## Similarity Metrics (Cosine / Dot / L2)

### Cosine similarity
Cosine similarity measures **direction**, not magnitude:

- same direction → similarity close to **1**
- orthogonal → around **0**
- opposite direction → close to **-1**

Why it’s popular:
- If you scale a vector up/down, cosine stays the same (direction unchanged).
- Many embedding pipelines normalize vectors to unit length, making cosine ≈ dot product.

---

## HNSW (Hierarchical Navigable Small World)

HNSW is one of the most common ANN structures because it’s fast and accurate.

### Representation
![HNSW](https://github.com/user-attachments/assets/9d0faf4c-6d5e-486c-b925-59b3aa96a819)

### Idea
HNSW builds a **multi-layer graph**:

- **Bottom layer** contains the full graph **G** (all vertices = vectors)
- Each upper layer is a smaller subgraph (fewer vertices, fewer edges)
- Upper layers act like “shortcuts” to quickly reach the right region of the space

### How search works (high-level)
1. Start at an entry point on the **top layer**
2. Greedily move to neighbors that are closer to the query
3. When no better neighbor exists, you found a **local minimum** on that layer
4. Drop down one layer and continue, using that local minimum as the new starting point
5. Repeat until the bottom layer
6. Return the closest node(s) (top-K)

### Intuition (map analogy)
Instead of searching for a street on one giant detailed map:

1. Map of countries → find **Russia**
2. Map of cities → find **Saint Petersburg**
3. Map of streets → find **Polytechnicheskaya 29**

This speeds up search and reduces the chance of ending up in the wrong “region”.

---

## DISKANN (DiskANN) — Disk-based Indexing

DISKANN is aimed at very large datasets where you can’t keep everything in RAM.

It combines two key ideas:

1. **Vamana Graph** — a disk-friendly graph index that connects points for efficient navigation  
2. **Product Quantization (PQ)** — compresses vectors to reduce memory and speed approximate distance computations

### Vamana Graph
The Vamana graph is central to DiskANN’s disk-based strategy.  
It can work with very large datasets because the index does not need to fully reside in memory.

![Vamana](https://github.com/user-attachments/assets/83b6e439-d163-4735-85ca-bb286dc617c1)

### Product Quantization (PQ)
PQ compresses vectors so that:

- distance computations can be done faster on compact codes
- memory usage drops significantly
- you can first do a fast approximate pass, then (optionally) refine using exact vectors

---

## Large Data: Why Chunking Is Required (Recap)

If you embed an entire book as one vector, retrieval becomes too broad.

Instead:
- split content into **chunks**
- embed each chunk
- store each chunk separately

Then queries retrieve the most relevant **paragraphs/sections**, not the whole book.

---

## Non-text Data (Images, Audio, Files)

For files, the vector is only a **search key**, not a reversible encoding.

Typical storage:
- `uuid`
- `file_link` / `storage_key`
- `vector`
- metadata

Important:
- you generally **cannot reconstruct the original file from the vector**
- the raw file must be stored separately (object storage / filesystem)


  


  
