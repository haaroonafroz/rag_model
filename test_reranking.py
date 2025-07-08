#!/usr/bin/env python3
"""
Test script to verify reranking functionality
"""

from src.rag_core import initialize_retriever
import time

def test_reranking():
    """Test reranking vs. no reranking on sample queries"""
    
    print("🧪 Testing Reranking vs. Standard Retrieval")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "What are the main requirements for financial institutions?",
        "How should ICT incidents be reported?",
        "What is the role of digital operational resilience?",
        "Who is responsible for monitoring compliance?"
    ]
    
    # Initialize retrievers
    print("📚 Initializing retrievers...")
    retriever_with_rerank, _ = initialize_retriever(use_reranking=True)
    retriever_without_rerank, _ = initialize_retriever(use_reranking=False)
    
    print(f"✅ Reranking enabled: {retriever_with_rerank.use_reranking}")
    print(f"🔤 Reranking model: {retriever_with_rerank.reranking_model is not None}")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        print("-" * 50)
        
        # Test with reranking
        start_time = time.time()
        docs_rerank = retriever_with_rerank.get_relevant_documents(query)
        rerank_time = time.time() - start_time
        
        # Test without reranking  
        start_time = time.time()
        docs_standard = retriever_without_rerank.get_relevant_documents(query)
        standard_time = time.time() - start_time
        
        print(f"⏱️  Timing:")
        print(f"   Standard: {standard_time:.3f}s")
        print(f"   Reranking: {rerank_time:.3f}s")
        print(f"   Overhead: +{(rerank_time - standard_time)*1000:.1f}ms")
        
        print(f"\n📊 Top 3 Results Comparison:")
        
        for j in range(min(3, len(docs_rerank), len(docs_standard))):
            print(f"\nRank #{j+1}:")
            
            # Standard result
            std_text = docs_standard[j].page_content[:100] + "..."
            print(f"   📈 Standard: {std_text}")
            
            # Reranked result
            rerank_doc = docs_rerank[j]
            rerank_text = rerank_doc.page_content[:100] + "..."
            metadata = getattr(rerank_doc, 'metadata', {})
            rerank_score = metadata.get('rerank_score', 'N/A')
            original_rank = metadata.get('original_rank', 'N/A')
            
            print(f"   🎯 Reranked: {rerank_text}")
            if rerank_score != 'N/A':
                print(f"      Score: {rerank_score:.4f}, Original Rank: #{original_rank}")
        
        print()

def test_reranking_scores():
    """Test reranking score distribution"""
    
    print("\n🎯 Reranking Score Analysis")
    print("=" * 40)
    
    retriever, _ = initialize_retriever(use_reranking=True)
    
    if not retriever.use_reranking:
        print("❌ Reranking not available")
        return
    
    query = "What are the requirements for financial entities?"
    docs = retriever.get_relevant_documents(query)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(docs)} documents")
    print("\nScore Distribution:")
    
    for i, doc in enumerate(docs):
        metadata = getattr(doc, 'metadata', {})
        score = metadata.get('rerank_score', 0)
        original_rank = metadata.get('original_rank', i+1)
        text_preview = doc.page_content[:60] + "..."
        
        print(f"  {i+1:2d}. Score: {score:6.4f} | Orig: #{original_rank:2d} | {text_preview}")

if __name__ == "__main__":
    try:
        test_reranking()
        test_reranking_scores()
        print("\n🎉 Reranking tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 