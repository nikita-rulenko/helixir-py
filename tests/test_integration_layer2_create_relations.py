#!/usr/bin/env python3
"""Layer 2: Тест MemoryIntegrator._create_relations()."""

import asyncio
from helixir.core.client import HelixDBClient
from helixir.core.config import HelixMemoryConfig
from helixir.toolkit.mind_toolbox.memory.integrator import MemoryIntegrator, MemoryRelation


async def test_create_relations():
    """Тестирует _create_relations напрямую."""
    config = HelixMemoryConfig(
        base_url="http://192.168.50.11:6969",
        timeout=30.0,
    )
    client = HelixDBClient(config)
    
    integrator = MemoryIntegrator(
        client=client,
        embedding_gen=None,  # Не нужен
        reasoning_engine=None,
    )
    
    try:
        # Используем существующие memory_id из БД
        source_id = "mem_001"
        target_id = "mem_tech_stack"
        
        print(f"🔍 Создаем тестовые relations...")
        print(f"   Source: {source_id}")
        print(f"   Target: {target_id}")
        
        # Создаем разные типы relations
        relations = [
            MemoryRelation(
                target_id=target_id,
                relation_type="MEMORY_RELATION",
                confidence=0.85,
                reasoning="Test relation via MemoryIntegrator",
            ),
        ]
        
        print(f"\n🔍 Вызываем _create_relations для {len(relations)} relations...")
        created_count = await integrator._create_relations(
            source_id=source_id,
            relations=relations,
        )
        
        print(f"\n✅ Создано {created_count} relations!")
        
        # Проверяем что relation создан - используем getMemoryLogicalConnections
        print(f"\n🔍 Проверяем созданные связи...")
        result = await client.execute_query(
            "getMemoryLogicalConnections",
            {"memory_id": source_id}
        )
        
        relation_out = result.get("relation_out", [])
        print(f"   Исходящих MEMORY_RELATION: {len(relation_out)}")
        
        if relation_out:
            print(f"   Первая связь:")
            first = relation_out[0]
            print(f"      target: {first.get('memory_id', 'N/A')}")
            print(f"      content: {first.get('content', 'N/A')[:80]}...")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_create_relations())

