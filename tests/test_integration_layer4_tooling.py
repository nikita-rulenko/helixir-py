#!/usr/bin/env python3
"""Layer 4: Тест ToolingManager.add_memory() через HelixirClient."""

import asyncio
from helixir.core.helixir_client import HelixirClient
from helixir.core.config import HelixMemoryConfig
from helixir.core.client import HelixDBClient


async def test_tooling_add_memory():
    """Тестирует add_memory через HelixirClient (который использует ToolingManager)."""
    
    # Конфигурация - ВСЁ из конфига, никаких manual инициализаций!
    config = HelixMemoryConfig(
        base_url="http://192.168.50.11:6969",
        timeout=30.0,
        llm_provider="ollama",  # Используем Ollama (доступен локально)
        llm_model="gemma2",
        embedding_url="http://192.168.50.2:11434",
        embedding_model="nomic-embed-text",
    )
    
    # HelixirClient инициализирует ВСЁ под капотом
    client = HelixirClient(config)
    
    try:
        message = "HelixDB memory integration pipeline automatically links new memories with existing ones via reasoning edges"
        user_id = "unknown"
        
        print(f"📝 Layer 4: HelixirClient.add() -> ToolingManager.add_memory()")
        print(f"   message: {message}")
        print(f"   user_id: {user_id}")
        
        print(f"\n🔍 Вызываем client.add()...")
        print(f"   (Full pipeline: LLM extract -> add DB -> integrate -> create relations)")
        
        result = await client.add(
            message=message,
            user_id=user_id,
        )
        
        print(f"\n✅ client.add() завершен!")
        print(f"   Added: {len(result.get('added', []))}")
        print(f"   Updated: {len(result.get('updated', []))}")
        print(f"   Deleted: {len(result.get('deleted', []))}")
        print(f"   Skipped: {result.get('skipped', 0)}")
        
        # Проверяем созданные relations
        added_ids = result.get('added', [])
        if added_ids:
            # Используем прямой доступ к БД для проверки
            db_client = HelixDBClient(config)
            
            for memory_id in added_ids:
                print(f"\n🔍 Проверяем reasoning связи для {memory_id}...")
                
                connections = await db_client.execute_query(
                    "getMemoryLogicalConnections",
                    {"memory_id": memory_id}
                )
                
                relation_out = connections.get("relation_out", [])
                implies_out = connections.get("implies_out", [])
                because_out = connections.get("because_out", [])
                contradicts_out = connections.get("contradicts_out", [])
                
                total = len(relation_out) + len(implies_out) + len(because_out) + len(contradicts_out)
                
                print(f"   Всего исходящих reasoning связей: {total}")
                print(f"      MEMORY_RELATION: {len(relation_out)}")
                print(f"      IMPLIES: {len(implies_out)}")
                print(f"      BECAUSE: {len(because_out)}")
                print(f"      CONTRADICTS: {len(contradicts_out)}")
                
                if total > 0:
                    print(f"\n   ✅ Layer 4 РАБОТАЕТ! Integration через ToolingManager!")
                    if relation_out:
                        print(f"   Первые 2 связи:")
                        for i, rel in enumerate(relation_out[:2], 1):
                            print(f"      {i}. -> {rel.get('memory_id', 'N/A')[:40]}")
                            print(f"         {rel.get('content', 'N/A')[:60]}...")
                else:
                    print(f"   ⚠️  Reasoning связи не созданы")
        else:
            print(f"\n⚠️  Воспоминания не добавлены (SKIP/UPDATE)")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_tooling_add_memory())

