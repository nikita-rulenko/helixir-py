#!/usr/bin/env python3
"""Layer 1: Тест MemoryIntegrator._find_similar_by_embedding()."""

import asyncio
from helixir.core.client import HelixDBClient
from helixir.core.config import HelixMemoryConfig
from helixir.llm.embeddings import EmbeddingGenerator
from helixir.toolkit.mind_toolbox.memory.integrator import MemoryIntegrator


async def test_find_similar():
    """Тестирует _find_similar_by_embedding напрямую."""
    config = HelixMemoryConfig(
        base_url="http://192.168.50.11:6969",
        timeout=30.0,
    )
    client = HelixDBClient(config)
    
    embedder = EmbeddingGenerator(
        ollama_url="http://192.168.50.2:11434",
        model="nomic-embed-text",
    )
    
    integrator = MemoryIntegrator(
        client=client,
        embedding_gen=embedder,
        reasoning_engine=None,  # Не нужен для этого теста
        similarity_threshold=0.5,  # Низкий порог для теста
        max_similar=5,
    )
    
    try:
        print("🔍 Генерируем эмбеддинг для поиска...")
        query = "HelixDB vector graph database"
        query_embedding = await embedder.generate(query)
        print(f"✅ Эмбеддинг готов: {len(query_embedding)} dimensions")
        
        print("\n🔍 Вызываем _find_similar_by_embedding...")
        similar = await integrator._find_similar_by_embedding(
            query_embedding=query_embedding,
            user_id="unknown",  # Реальный user_id из БД
            exclude_id=None,
        )
        
        print(f"\n✅ Найдено {len(similar)} похожих воспоминаний")
        
        if similar:
            print("\n📝 Первые 3 результата:")
            for i, sim in enumerate(similar[:3], 1):
                print(f"\n   {i}. memory_id: {sim.memory_id}")
                print(f"      content: {sim.content[:80]}...")
                print(f"      similarity: {sim.similarity_score:.3f}")
                print(f"      created_at: {sim.created_at}")
        else:
            print("\n⚠️  Нет похожих воспоминаний (возможно нет данных для test_user)")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_find_similar())

