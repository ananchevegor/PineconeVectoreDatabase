# Information relevant to the topic
First of all, this is all for myself.
These will just be notes about what I’ve understood, etc.

```

1. Vector databases store information in vectors and perform search “by meaning” (semantic search). This means that, unlike a regular indexed database where you search by an id (or other parameters) that must STRICTLY match a record, in a vector database such an exact match is not required.

2. The search is performed using the cosine metric (the cosine between vectors). You can think of it as the cosine of the angle between two hypotenuses formed by the distances between three vertices of a graph (the initial vector – the one we provide – and its two nearest neighbors). The closer the cosine is to 1, the higher the similarity between the vectors.

3. Why vectors? Because they have direction! A vector that is twice as long will have the same cosine as a vector that is half as long. A vector in the opposite direction will have a cosine of -1.

```

# Embedding

Embedding is an algorithm for transforming data into a vector.

### Types of Embedding

  ```

   1. Dense – a vector whose elements have non-zero values. These vectors usually have lower dimensionality and are well suited for search.

   2. Sparse – a vector that has many zero-valued elements. These vectors usually have a much higher dimensionality (10k–1,000k+). They are not typically used for training neural networks, but are used in search systems in browsers.

  ```

### What is this output vector?

  ```python
    v = [0.13, ..., 0.875] – of dimensionality n
  ```

  means a point in an n-dimensional space, where each axis is a direction

# What to do if the data is large.

Let’s assume we need to load War and Peace by Tolstoy into a database. We can’t just take and load the whole thing in one go, at least because it would lose its meaning if we search by a single query against the ENTIRE book.
For example, if someone types “What was the oak like in the novel?” or “How did Prince Bolkonsky die?”, the answer would be the whole book. We don’t need that for further AI data processing.

We have to split the text into chunks.
A chunk is data of a certain size.

There are different ways to split it:

By meaning (semantic chunks)

By number of words (for example, 500 words)

By chapters (text.split("\n\n"))

Custom – by any other criterion

This way, for two different queries we will get two nearest vertices of the graph. They will not be the whole book. They will be individual sentences or paragraphs.



## What if the data is not text
Let’s assume we want to store a vector of a file (an image, audio file, ...).
When we convert text into a vector, we basically store: uuid, chunk_text, vector – essentially, we use the vector to search for chunk_text – it has no encoding, it’s just regular text.
With files we store: uuid, file_link, vector – the same idea, except we also have to store the actual files somewhere else. We cannot convert back from the vector into the original data.
## Building an index in a database
How it works. How search works. Algorithms.
HNSW (Hierarchical Navigable Small World) – a search algorithm based on a hierarchy (layers) with graphs that approximate the vertex–vector at the last layer.
Representation:
  ![image](https://github.com/user-attachments/assets/9d0faf4c-6d5e-486c-b925-59b3aa96a819)


The graph G is a graph consisting of all vertices and edges present in the database (uuid + embedding).
On the top layer there is the “smallest” subgraph of graph G. On the bottom layer there is the full graph G. The top layer contains n vertices and n − 1 edges. The value of n is optional, the selection is random.

On the top layer there is a local minimum (the vertex closest to the query).

On the next layer there is a selection of the next n vertices that are neighbors of the local minimum from the previous layer.

On the last layer we have the entire graph. Knowing the local minimum of the previous layer, we start from this vertex. The process is repeated and again a local minimum is found. The answer is the local minimum and its neighbors of the lower layer.

A vivid example:
Let’s assume that on my map there are no country and city names, but there are street names. I need to find Polytechnicheskaya street 29 in Saint Petersburg. Instead of looking for this street on a single map, I first take a map with countries and find Russia. Then I take a map with cities and find Saint Petersburg. And on the final map with streets I find the desired street.
This greatly speeds up the search and also eliminates errors where we might end up at Polytechnicheskaya street 29 in Moscow or any other city (if such a street exists there).

DISKANN is an index-building algorithm that includes two technologies:

Vamana Graph – a disk-based graph index that connects data points (or vectors) for efficient navigation during search.

Product Quantization (PQ) – an in-memory compression method that reduces the size of vectors, allowing fast computation of approximate distances between them.

Vamana graph
The Vamana graph plays a central role in DISKANN’s disk-based strategy. It can work with very large datasets because it does not need to fully reside in memory during or after construction.
  
![image](https://github.com/user-attachments/assets/83b6e439-d163-4735-85ca-bb286dc617c1)

  


  
