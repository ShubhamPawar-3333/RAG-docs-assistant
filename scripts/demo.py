"""
Quick Demo Script for RAG Pipeline

A simple script to demonstrate the RAG pipeline in action.
Run this after setting up your GOOGLE_API_KEY in .env
"""

from src.rag import (
    load_documents,
    chunk_documents,
    get_embeddings,
    create_vector_store,
    create_rag_pipeline,
)

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



def main():
    print("🚀 DocuMind AI - RAG Demo")
    print("=" * 50)
    
    # Step 1: Load and process documents
    print("\n📄 Loading documents...")
    docs = load_documents("data/sample_docs")
    print(f"   Loaded {len(docs)} document(s)")
    
    # Step 2: Chunk documents
    print("\n✂️  Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"   Created {len(chunks)} chunks")
    
    # Step 3: Create vector store and add documents
    print("\n🗄️  Setting up vector store...")
    embeddings = get_embeddings()
    store = create_vector_store(embeddings, collection_name="demo_docs")
    store.add_documents(chunks)
    print(f"   Indexed {len(chunks)} chunks")
    
    # Step 4: Create RAG pipeline
    print("\n🔗 Creating RAG pipeline...")
    pipeline = create_rag_pipeline(collection_name="demo_docs")
    print("   Pipeline ready!")
    
    # Step 5: Interactive Q&A
    print("\n" + "=" * 50)
    print("💬 Ask questions about your documents")
    print("   Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n🔍 Searching and generating answer...")
            result = pipeline.query(question, include_sources=True)
            
            print(f"\n💬 Answer:\n{result['answer']}")
            
            if result.get('sources'):
                print("\n📚 Sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    fname = source['metadata'].get('file_name', 'unknown')
                    score = source.get('score', 0)
                    print(f"   {i}. {fname} (relevance: {score:.2f})")
                    
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
