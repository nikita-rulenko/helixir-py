#!/usr/bin/env python3
"""Layer 3: Тест MemoryIntegrator.integrate_memory() полный цикл."""

import asyncio
from datetime import UTC, datetime
from helixir.core.client import HelixDBClient
from helixir.core.config import HelixMemoryConfig
from helixir.llm.embeddings import EmbeddingGenerator
from helixir.toolkit.mind_toolbox.memory.integrator import MemoryIntegrator
from helixir.toolkit.mind_toolbox.memory.models import Memory


async def test_integrate_memory():
    """Тестирует полный цикл integrate_memory."""
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
        reasoning_engine=None,  # Пока без LLM reasoning
        similarity_threshold=0.7,
        max_similar=5,
        enable_reasoning=False,  # Отключаем LLM для простоты
    )
    
    try:
        # ШАГ 1: Создаем Memory в БД
        memory_id = "test_integration_mem_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        content = "Testing HelixDB memory integration with graph reasoning and vector search"
        
        print(f"📝 ШАГ 1: Добавляем память в БД")
        print(f"   memory_id: {memory_id}")
        print(f"   content: {content}")
        
        # Генерируем эмбеддинг
        print(f"\n🔍 Генерируем эмбеддинг...")
        query_embedding = await embedder.generate(content)
        print(f"✅ Эмбеддинг готов: {len(query_embedding)} dimensions")
        
        # Добавляем Memory node
        memory_result = await client.execute_query(
            "addMemory",
            {
                "memory_id": memory_id,
                "content": content,
                "memory_type": "fact",
                "certainty": 90,
                "importance": 80,
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "context_tags": "",
                "source": "test",
                "metadata": "{}",
            }
        )
        memory_node = memory_result.get("memory")
        internal_id = memory_node.get("id")  # Internal UUID
        print(f"✅ Memory node создан в БД (internal_id: {internal_id})")
        
        # Добавляем MemoryEmbedding и линкуем
        await client.execute_query(
            "addMemoryEmbedding",
            {
                "memory_id": internal_id,  # Используем internal ID!
                "vector_data": query_embedding,
                "content": content,
                "embedding_model": "nomic-embed-text",
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        print(f"✅ MemoryEmbedding создан и залинкован")
        
        # ШАГ 2: Создаем Memory object для integrator
        test_memory = Memory(
            memory_id=memory_id,
            user_id="unknown",
            agent_id=None,
            content=content,
            memory_type="fact",
            created_at=datetime.now(UTC),
        )
        
        print(f"\n📝 ШАГ 2: Вызываем integrate_memory (полный цикл)")
        print(f"   Будет искать похожие и создавать связи...")
        
        result = await integrator.integrate_memory(
            memory=test_memory,
            query_embedding=query_embedding,
        )
        
        print(f"\n✅ Интеграция завершена!")
        print(f"   Similar found: {result.similar_found}")
        print(f"   Relations created: {result.relations_created}")
        print(f"   Superseded: {len(result.superseded_memories)}")
        print(f"   Time: {result.integration_time_ms:.2f}ms")
        
        if result.similar_found > 0:
            print(f"\n📊 Найдено {result.similar_found} похожих воспоминаний")
            print(f"   Создано {result.relations_created} связей")
            
            # ШАГ 3: Проверяем что связи реально созданы
            print(f"\n📝 ШАГ 3: Проверяем созданные связи в БД")
            connections = await client.execute_query(
                "getMemoryLogicalConnections",
                {"memory_id": memory_id}
            )
            
            relation_out = connections.get("relation_out", [])
            print(f"   Исходящих MEMORY_RELATION: {len(relation_out)}")
            
            if relation_out:
                print(f"   ✅ Связи созданы:")
                for i, rel in enumerate(relation_out[:3], 1):
                    print(f"      {i}. -> {rel.get('memory_id', 'N/A')[:40]}")
                    print(f"         {rel.get('content', 'N/A')[:60]}...")
            else:
                print(f"   ❌ Связи НЕ созданы в БД!")
        else:
            print(f"\n⚠️  Похожих воспоминаний не найдено")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_integrate_memory())

