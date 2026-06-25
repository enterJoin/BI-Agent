# springgraph

`springgraph` scans a Java Spring Boot directory and writes a local code graph to PostgreSQL.

The entry point accepts a single path. That path can be one Spring Boot project or a workspace containing multiple Spring Boot projects:

```powershell
springgraph index G:\path\to\java-workspace
springgraph status G:\path\to\java-workspace
springgraph routes G:\path\to\java-workspace
springgraph resources G:\path\to\java-workspace
```

Run the FastAPI service with:

```powershell
springgraph-api
```

Then call:

```http
POST /api/refine
{
  "project_path": "F:\\path\\to\\gulimall"
}
```

and:

```http
POST /api/vector-search
{
  "project_id": "project:f180a56f5c2d6c1c9d064558ba21cdfa",
  "query": "订单系统和购物车系统有什么交集？",
  "limit": 10
}
```

You can also pass `project_path` instead of `project_id`:

```http
POST /api/vector-search
{
  "project_path": "F:\\path\\to\\gulimall",
  "query": "订单系统和购物车系统有什么交集？",
  "limit": 10
}
```

The refinement endpoint writes relational graph data and chunk embeddings. The vector search endpoint only retrieves from the embedding store and returns the closest chunks for downstream RAG use. One of `project_id` or `project_path` is required; if both are omitted, the API returns `400`.

Project business knowledge can be added as Markdown under a top-level `library` directory:

```text
java-workspace/
  library/
    gulimall-business-knowledge.md
```

During refinement, `library/**/*.md` files are written to the same chunk and embedding tables as code-derived chunks. Each Markdown heading section is stored as one parent knowledge chunk plus smaller child chunks. Child chunks include neighboring text overlap so vector retrieval can match precise keywords without losing nearby context; metadata keeps the parent document ID, parent title, full parent content, chunk role, chunk index, chunk count, and overlap setting.
