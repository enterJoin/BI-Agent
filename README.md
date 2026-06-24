# springgraph

`springgraph` scans a Java Spring Boot directory and writes a local code graph to PostgreSQL.

The entry point accepts a single path. That path can be one Spring Boot project or a workspace containing multiple Spring Boot projects:

```powershell
springgraph index G:\path\to\java-workspace
springgraph status G:\path\to\java-workspace
springgraph routes G:\path\to\java-workspace
springgraph resources G:\path\to\java-workspace
```

The first version stores structured relational data only. pgvector and RAG are planned later.

