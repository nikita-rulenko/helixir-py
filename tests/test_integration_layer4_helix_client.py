#!/usr/bin/env python3
"""Layer 4: Тест HelixirClient.add() с Memory Integration."""

import asyncio
import os
from helixir.core.helixir_client import HelixirClient
from helixir.core.config import HelixMemoryConfig
from helixir.core.client import HelixDBClient

# Set ENV for testing (simulates mcp.json)
os.environ["HELIX_LLM_API_KEY"] = "csk-548wv288yv58928r5mwk322m6vw64rmrk22rt8ymdncv3tyx"


async def test_layer_4_1_init():
    """Слой 4.1: Инициализация HelixirClient из конфига."""
    print("=" * 60)
    print("LAYER 4.1: Инициализация HelixirClient")
    print("=" * 60)
    
    # Загружаем конфиг с правильным приоритетом: ENV > YAML > Defaults
    # Это позволяет mcp.json переопределять настройки
    try:
        config = HelixMemoryConfig.from_yaml()
        print(f"✅ Loaded config from config.yaml (with ENV overrides)")
    except FileNotFoundError:
        print(f"⚠️  config.yaml not found, using ENV + Defaults")
        config = HelixMemoryConfig()
    
    print(f"\n📝 Конфигурация:")
    print(f"   DB: {config.base_url}")
    print(f"   LLM: {config.llm_provider}/{config.llm_model}")
    print(f"   LLM API Key: {'***' + config.llm_api_key[-8:] if config.llm_api_key else 'None'}")
    print(f"   LLM URL: {config.llm_base_url}")
    print(f"   Embeddings: {config.embedding_model} @ {config.embedding_url}")
    
    print(f"\n🔍 Инициализируем HelixirClient...")
    client = HelixirClient(config)
    
    print(f"✅ HelixirClient создан")
    print(f"   - db: {type(client.db).__name__}")
    print(f"   - llm_provider: {type(client.llm_provider).__name__}")
    print(f"   - extractor: {type(client.extractor).__name__}")
    print(f"   - embedder: {type(client.embedder).__name__}")
    print(f"   - tooling: {type(client.tooling).__name__}")
    
    return client, config


async def _layer_4_2_add(client):
    """Слой 4.2: Вызов client.add()."""
    print("\n" + "=" * 60)
    print("LAYER 4.2: client.add()")
    print("=" * 60)
    
    message = "HelixDB automatically creates reasoning links between related memories using graph traversal"
    user_id = "unknown"
    
    print(f"📝 Добавляем воспоминание:")
    print(f"   message: {message[:80]}...")
    print(f"   user_id: {user_id}")
    
    print(f"\n🔍 Сначала проверим LLM extraction отдельно...")
    try:
        extraction = await client.extractor.extract(
            text=message,
            user_id=user_id,
            extract_entities=True,
            extract_relations=True,
        )
        print(f"   LLM extraction результат:")
        print(f"      memories: {len(extraction.memories)}")
        print(f"      entities: {len(extraction.entities)}")
        print(f"      relations: {len(extraction.relations)}")
        
        if extraction.memories:
            print(f"   Первое воспоминание:")
            print(f"      content: {extraction.memories[0].content[:80]}...")
        else:
            print(f"   ⚠️  LLM НЕ извлек memories!")
    except Exception as e:
        print(f"   ❌ LLM extraction упал: {e}")
    
    print(f"\n🔍 Теперь вызываем client.add() (full pipeline)...")
    result = await client.add(
        message=message,
        user_id=user_id,
    )
    
    print(f"\n✅ client.add() завершен!")
    print(f"   Added: {result.get('added', [])}")
    print(f"   Updated: {result.get('updated', [])}")
    print(f"   Deleted: {result.get('deleted', [])}")
    print(f"   Skipped: {result.get('skipped', 0)}")
    
    added_ids = result.get('added', [])
    if not added_ids:
        print(f"\n⚠️  WARNING: Ничего не добавлено! Возможно SKIP или UPDATE")
        return None
    
    return added_ids[0]  # Возвращаем первый добавленный ID


async def _layer_4_3_check_relations(memory_id, config):
    """Слой 4.3: Проверка reasoning связей."""
    print("\n" + "=" * 60)
    print("LAYER 4.3: Проверка reasoning связей")
    print("=" * 60)
    
    db_client = HelixDBClient(config)
    
    print(f"🔍 Проверяем связи для {memory_id}...")
    connections = await db_client.execute_query(
        "getMemoryLogicalConnections",
        {"memory_id": memory_id}
    )
    
    relation_out = connections.get("relation_out", [])
    implies_out = connections.get("implies_out", [])
    because_out = connections.get("because_out", [])
    contradicts_out = connections.get("contradicts_out", [])
    
    total = len(relation_out) + len(implies_out) + len(because_out) + len(contradicts_out)
    
    print(f"\n📊 Reasoning связи:")
    print(f"   MEMORY_RELATION: {len(relation_out)}")
    print(f"   IMPLIES: {len(implies_out)}")
    print(f"   BECAUSE: {len(because_out)}")
    print(f"   CONTRADICTS: {len(contradicts_out)}")
    print(f"   TOTAL: {total}")
    
    if total > 0:
        print(f"\n✅ LAYER 4 РАБОТАЕТ! Reasoning связи созданы!")
        if relation_out:
            print(f"\n   Первые 2 MEMORY_RELATION:")
            for i, rel in enumerate(relation_out[:2], 1):
                print(f"      {i}. -> {rel.get('memory_id', 'N/A')[:40]}")
                print(f"         {rel.get('content', 'N/A')[:70]}...")
        return True
    else:
        print(f"\n❌ LAYER 4 FAILED: Reasoning связи НЕ созданы")
        return False


async def test_layer_4_full():
    """Полный тест Layer 4."""
    try:
        # Слой 4.1
        client, config = await test_layer_4_1_init()
        
        # Слой 4.2
        memory_id = await _layer_4_2_add(client)
        if not memory_id:
            print("\n❌ Layer 4 прерван: не удалось добавить воспоминание")
            return
        
        # Слой 4.3
        success = await _layer_4_3_check_relations(memory_id, config)
        
        # Итог
        print("\n" + "=" * 60)
        if success:
            print("✅ LAYER 4 COMPLETE: HelixirClient.add() + Integration работает!")
        else:
            print("❌ LAYER 4 FAILED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Ошибка в Layer 4: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_layer_4_full())

