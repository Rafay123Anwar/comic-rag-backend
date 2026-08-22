-- ==============================================================================
-- COMIC-RAG: Supabase pgvector Database Schema & Search RPC
-- ==============================================================================
-- Run this script in the Supabase Dashboard -> SQL Editor
-- (https://supabase.com/dashboard/project/_/sql)

-- 1. Enable the pgvector extension for high-performance vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the comic_page_chunks table for storing chunk text & 1024-dim embeddings
CREATE TABLE IF NOT EXISTS public.comic_page_chunks (
    id TEXT PRIMARY KEY,
    comic_id TEXT NOT NULL,
    page_number INTEGER NOT NULL DEFAULT 1,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(1024),
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- 3. Create indexes for fast lookup and filtering by comic_id and page_number
CREATE INDEX IF NOT EXISTS idx_comic_page_chunks_comic_id 
    ON public.comic_page_chunks (comic_id);

CREATE INDEX IF NOT EXISTS idx_comic_page_chunks_page 
    ON public.comic_page_chunks (comic_id, page_number);

-- 4. Create HNSW vector index for high-speed approximate cosine similarity search
CREATE INDEX IF NOT EXISTS idx_comic_page_chunks_embedding_hnsw 
    ON public.comic_page_chunks 
    USING hnsw (embedding vector_cosine_ops);

-- 5. Enable Row Level Security (RLS) if desired, or allow authenticated/service access
ALTER TABLE public.comic_page_chunks ENABLE ROW LEVEL SECURITY;

-- Allow public read/write access with service role or anon key (for backend API)
DROP POLICY IF EXISTS "Allow backend full access" ON public.comic_page_chunks;
CREATE POLICY "Allow backend full access" 
    ON public.comic_page_chunks 
    FOR ALL 
    USING (true) 
    WITH CHECK (true);

-- 6. RPC Function: match_comic_chunks
-- Performs cosine vector distance search scoped strictly to a specific comic_id
CREATE OR REPLACE FUNCTION public.match_comic_chunks (
    query_embedding vector(1024),
    match_threshold double precision DEFAULT 1.0,
    match_count integer DEFAULT 8,
    p_comic_id text DEFAULT NULL
)
RETURNS TABLE (
    id text,
    comic_id text,
    page_number integer,
    chunk_index integer,
    content text,
    metadata jsonb,
    similarity double precision,
    distance double precision
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        cpc.id,
        cpc.comic_id,
        cpc.page_number,
        cpc.chunk_index,
        cpc.content,
        cpc.metadata,
        (1.0 - (cpc.embedding <=> query_embedding)) AS similarity,
        (cpc.embedding <=> query_embedding) AS distance
    FROM public.comic_page_chunks cpc
    WHERE (p_comic_id IS NULL OR cpc.comic_id = p_comic_id)
      AND (cpc.embedding <=> query_embedding) <= match_threshold
    ORDER BY cpc.embedding <=> query_embedding ASC
    LIMIT match_count;
$$;
