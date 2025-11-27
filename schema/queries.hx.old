// =============================================================================
// HelixDB Unified Queries - Single Source of Truth
// =============================================================================
// Generated: 2025-11-18
// Contains: Cognitive Engine (26 queries) + RAG System (8 queries)
// Total: 34 queries
// =============================================================================

// =============================================================================
// SECTION 1: COGNITIVE ENGINE QUERIES (Level 0-5.5)
// =============================================================================

// Level 0: User Management
QUERY addUser(user_id: String, name: String) =>
  user <- AddN<User>({ user_id: user_id, name: name })
  RETURN user

QUERY getUser(user_id: String) =>
  user <- N<User>::WHERE(_::{user_id}::EQ(user_id))::FIRST
  RETURN user

// Level 1: Memory CRUD
QUERY addMemory(memory_id: String, content: String, memory_type: String, certainty: I64, importance: I64, created_at: String, updated_at: String, context_tags: String, source: String, metadata: String) =>
  memory <- AddN<Memory>({ memory_id: memory_id, content: content, memory_type: memory_type, certainty: certainty, importance: importance, created_at: created_at, updated_at: updated_at, context_tags: context_tags, source: source, metadata: metadata })
  RETURN memory

QUERY getMemory(memory_id: String) =>
  memory <- N<Memory>::WHERE(_::{memory_id}::EQ(memory_id))::FIRST
  RETURN memory

QUERY getRecentMemories(limit: I64) =>
  memories <- N<Memory>::RANGE(0, limit)
  RETURN memories

// Level 2: Context Management
QUERY addContext(context_id: String, name: String, context_type: String, properties: String, parent_context: String) =>
  context <- AddN<Context>({ context_id: context_id, name: name, context_type: context_type, properties: properties, parent_context: parent_context })
  RETURN context

QUERY getContext(context_id: String) =>
  context <- N<Context>::WHERE(_::{context_id}::EQ(context_id))::FIRST
  RETURN context

QUERY getRecentContexts(limit: I64) =>
  contexts <- N<Context>::RANGE(0, limit)
  RETURN contexts

// Level 3: Update Operations
QUERY updateMemory(memory_id: String, content: String, certainty: I64, importance: I64, updated_at: String) =>
  memory <- N<Memory>::WHERE(_::{memory_id}::EQ(memory_id))::FIRST
  memory_internal_id <- memory::ID
  updated <- N<Memory>(memory_internal_id)::UPDATE({ content: content, certainty: certainty, importance: importance, updated_at: updated_at })
  RETURN updated

QUERY updateMemoryById(id: ID, content: String, certainty: I64, importance: I64, updated_at: String) =>
  updated <- N<Memory>(id)::UPDATE({ content: content, certainty: certainty, importance: importance, updated_at: updated_at })
  RETURN updated

// Level 4: Graph Relations
QUERY addMemoryRelation(source_id: String, target_id: String, relation_type: String, strength: I64, created_at: String, metadata: String) =>
  source <- N<Memory>::WHERE(_::{memory_id}::EQ(source_id))::FIRST
  target <- N<Memory>::WHERE(_::{memory_id}::EQ(target_id))::FIRST
  relation <- AddE<MEMORY_RELATION>({ relation_type: relation_type, strength: strength, created_at: created_at, metadata: metadata })::From(source::ID)::To(target::ID)
  RETURN relation

QUERY getRelatedMemories(memory_id: ID) =>
  memory <- N<Memory>(memory_id)
  related <- memory::Out<MEMORY_RELATION>
  RETURN related

QUERY addMemoryImplication(from_id: String, to_id: String, probability: I64, reasoning_id: String) =>
  from_memory <- N<Memory>::WHERE(_::{memory_id}::EQ(from_id))::FIRST
  to_memory <- N<Memory>::WHERE(_::{memory_id}::EQ(to_id))::FIRST
  implication <- AddE<IMPLIES>({ probability: probability, reasoning_id: reasoning_id })::From(from_memory::ID)::To(to_memory::ID)
  RETURN implication

QUERY addMemoryCausation(from_id: String, to_id: String, strength: I64, reasoning_id: String) =>
  from_memory <- N<Memory>::WHERE(_::{memory_id}::EQ(from_id))::FIRST
  to_memory <- N<Memory>::WHERE(_::{memory_id}::EQ(to_id))::FIRST
  causation <- AddE<BECAUSE>({ strength: strength, reasoning_id: reasoning_id })::From(from_memory::ID)::To(to_memory::ID)
  RETURN causation

QUERY addMemoryContradiction(from_id: String, to_id: String, resolution: String, resolved: I64, resolution_strategy: String) =>
  from_memory <- N<Memory>::WHERE(_::{memory_id}::EQ(from_id))::FIRST
  to_memory <- N<Memory>::WHERE(_::{memory_id}::EQ(to_id))::FIRST
  contradiction <- AddE<CONTRADICTS>({ resolution: resolution, resolved: resolved, resolution_strategy: resolution_strategy })::From(from_memory::ID)::To(to_memory::ID)
  RETURN contradiction

QUERY addMemoryToContext(memory_id: String, context_id: String, timestamp: String) =>
  memory <- N<Memory>::WHERE(_::{memory_id}::EQ(memory_id))::FIRST
  context <- N<Context>::WHERE(_::{context_id}::EQ(context_id))::FIRST
  link <- AddE<OCCURRED_IN>({ timestamp: timestamp })::From(memory::ID)::To(context::ID)
  RETURN link

QUERY getMemoryContext(memory_id: ID) =>
  memory <- N<Memory>(memory_id)
  context <- memory::Out<OCCURRED_IN>
  RETURN context

QUERY getContextMemories(context_id: ID) =>
  context <- N<Context>(context_id)
  memories <- context::In<OCCURRED_IN>
  RETURN memories

// Level 5: Vector Operations
QUERY addMemoryEmbedding(memory_id: ID, vector_data: [F64], content: String, embedding_model: String, created_at: String) =>
  embedding <- AddV<MemoryEmbedding>(vector_data, { content: content, created_at: created_at })
  link <- AddE<HAS_EMBEDDING>({ embedding_model: embedding_model })::From(memory_id)::To(embedding)
  RETURN embedding

QUERY addEntityEmbedding(entity_id: ID, vector_data: [F64], content: String, embedding_model: String) =>
  embedding <- AddV<EntityEmbedding>(vector_data, { name: content })
  link <- AddE<ENTITY_HAS_EMBEDDING>({ embedding_model: embedding_model })::From(entity_id)::To(embedding)
  RETURN embedding

QUERY searchSimilarMemories(query_vector: [F64], limit: I64) =>
  embeddings <- SearchV<MemoryEmbedding>(query_vector, limit)
  RETURN embeddings

QUERY vectorSearch(query_vector: [F64], user_id: String, limit: I64, min_score: F64) =>
  embeddings <- SearchV<MemoryEmbedding>(query_vector, limit)
  RETURN embeddings

QUERY searchSimilarEntities(query_vector: [F64], limit: I64) =>
  embeddings <- SearchV<EntityEmbedding>(query_vector, limit)
  RETURN embeddings

QUERY searchMemoriesByContext(query_vector: [F64], context_id: ID, limit: I64) =>
  context <- N<Context>(context_id)
  context_memories <- context::In<OCCURRED_IN>
  embeddings <- SearchV<MemoryEmbedding>(query_vector, limit)
  RETURN embeddings

QUERY searchRecentMemories(query_vector: [F64], limit: I64, cutoff_date: String) =>
  embeddings <- SearchV<MemoryEmbedding>(query_vector, limit)::WHERE(_::{created_at}::GTE(cutoff_date))
  RETURN embeddings

// Level 5.5: Chunking
QUERY addMemoryChunk(chunk_id: String, parent_memory_id: String, position: I64, content: String, token_count: I64, created_at: String) =>
  chunk <- AddN<MemoryChunk>({ chunk_id: chunk_id, parent_memory_id: parent_memory_id, position: position, content: content, token_count: token_count, created_at: created_at })
  parent <- N<Memory>::WHERE(_::{memory_id}::EQ(parent_memory_id))::FIRST
  link <- AddE<HAS_CHUNK>({ chunk_index: position })::From(parent)::To(chunk)
  RETURN chunk

// =============================================================================
// SECTION 2: RAG SYSTEM QUERIES
// =============================================================================

// Documentation Management
QUERY addDocPage(url: String, title: String, category: String, word_count: I64) =>
  page <- AddN<DocPage>({ url: url, title: title, category: category, word_count: word_count })
  RETURN page

QUERY addDocChunk(chunk_id: String, content: String, chunk_index: I64, word_count: I64, section_title: String, page_url: String) =>
  chunk <- AddN<DocChunk>({ chunk_id: chunk_id, content: content, chunk_index: chunk_index, word_count: word_count, section_title: section_title })
  page <- N<DocPage>::WHERE(_::{url}::EQ(page_url))::FIRST
  link <- AddE<PAGE_TO_CHUNK>({})::From(page)::To(chunk)
  RETURN chunk

QUERY addChunkEmbedding(chunk_id: String, vector_data: [F64]) =>
  chunk <- N<DocChunk>::WHERE(_::{chunk_id}::EQ(chunk_id))::FIRST
  embedding <- AddV<ChunkEmbedding>(vector_data, {})
  link <- AddE<CHUNK_TO_EMBEDDING>({})::From(chunk)::To(embedding)
  RETURN embedding

QUERY searchDocChunks(query_vector: [F64], limit: I64) =>
  embeddings <- SearchV<ChunkEmbedding>(query_vector, limit)
  RETURN embeddings

QUERY getDocChunksByPage(page_url: String) =>
  page <- N<DocPage>::WHERE(_::{url}::EQ(page_url))::FIRST
  chunks <- page::Out<PAGE_TO_CHUNK>
  RETURN chunks

QUERY addCodeExample(example_id: String, code: String, language: String, description: String) =>
  example <- AddN<CodeExample>({ example_id: example_id, code: code, language: language, description: description })
  RETURN example

QUERY linkChunkToExample(chunk_id: String, example_id: String) =>
  chunk <- N<DocChunk>::WHERE(_::{chunk_id}::EQ(chunk_id))::FIRST
  example <- N<CodeExample>::WHERE(_::{example_id}::EQ(example_id))::FIRST
  link <- AddE<CHUNK_HAS_EXAMPLE>({})::From(chunk)::To(example)
  RETURN link

QUERY searchConceptsByName(name: String) =>
  concepts <- N<Concept>::WHERE(_::{name}::EQ(name))
  RETURN concepts
