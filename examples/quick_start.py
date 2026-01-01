"""
Quick start example: Using RankSaga's optimized embedding model.

This example demonstrates how to use the RankSaga-optimized E5-v2 model
for information retrieval tasks.
"""
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np

def main():
    print("="*60)
    print("RankSaga Optimized Embedding Model - Quick Start")
    print("="*60)
    
    # Load the model
    print("\n1. Loading model...")
    print("   Model: RankSaga/ranksaga-optimized-e5-v2")
    
    try:
        model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        print("\n   💡 If model not found on Hugging Face, use local model:")
        print("      model = SentenceTransformer('path/to/local/model')")
        return
    
    # Example 1: Basic text encoding
    print("\n2. Encoding text...")
    sentences = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Deep learning uses neural networks with multiple layers.",
        "Machine learning enables computers to learn from data."
    ]
    
    embeddings = model.encode(sentences, show_progress_bar=True)
    print(f"   ✅ Encoded {len(sentences)} sentences")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # Example 2: Computing similarity
    print("\n3. Computing similarities...")
    query = "What is machine learning?"
    query_embedding = model.encode(query)
    
    # Compute similarities with all documents
    similarities = cos_sim(query_embedding, embeddings)[0]
    
    # Find most similar
    top_idx = similarities.argmax().item()
    top_score = similarities[top_idx].item()
    
    print(f"\n   Query: '{query}'")
    print(f"   Most similar document: '{sentences[top_idx]}'")
    print(f"   Similarity score: {top_score:.4f}")
    
    # Example 3: Information retrieval
    print("\n4. Information retrieval example...")
    
    # Documents corpus
    documents = [
        "Artificial intelligence is transforming healthcare through diagnostic tools.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Robotics combines AI with mechanical engineering for automation.",
    ]
    
    # Query
    search_query = "How does AI help with medical diagnosis?"
    
    # Encode
    doc_embeddings = model.encode(documents)
    query_emb = model.encode(search_query)
    
    # Retrieve top results
    similarities = cos_sim(query_emb, doc_embeddings)[0]
    top_k = 2
    
    # Get top-k indices
    top_indices = np.argsort(similarities.numpy())[::-1][:top_k]
    
    print(f"\n   Query: '{search_query}'")
    print(f"\n   Top {top_k} results:")
    for i, idx in enumerate(top_indices, 1):
        print(f"\n   {i}. Score: {similarities[idx]:.4f}")
        print(f"      Document: '{documents[idx]}'")
    
    # Example 4: Batch encoding
    print("\n5. Batch encoding example...")
    batch_queries = [
        "What is deep learning?",
        "How do neural networks work?",
        "What is the difference between AI and ML?"
    ]
    
    batch_embeddings = model.encode(batch_queries, batch_size=32, show_progress_bar=True)
    print(f"   ✅ Encoded {len(batch_queries)} queries in batch")
    print(f"   Shape: {batch_embeddings.shape}")
    
    print("\n" + "="*60)
    print("✅ Quick start complete!")
    print("="*60)
    print("\nFor more examples and documentation, see:")
    print("  - README.md")
    print("  - https://huggingface.co/RankSaga/ranksaga-optimized-e5-v2")
    print("  - https://ranksaga.com/blog/beir-benchmarking-ranksaga-optimization")


if __name__ == "__main__":
    main()

