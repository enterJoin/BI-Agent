# RAG Chat Frontend API

This document describes the chat-thread APIs, the vector search API, and the
streaming RAG ask API used by the frontend.

## Conventions

- Base URL examples use the current FastAPI app root, for example
  `http://localhost:8000`.
- Request bodies should use `camelCase` names. The backend also accepts the
  older `snake_case` names for compatibility where implemented.
- Response bodies currently use `snake_case` names, for example `thread_id`,
  `project_id`, and `user_id`.
- `userId` is optional. If the frontend does not pass it, the backend uses
  `"1"`.
- `threadId` is optional. If the frontend does not pass it, the backend creates
  a new id in the form `thread:<uuid-hex>`.
- `title` is optional. If omitted, the backend uses the first user question or
  vector query as the chat title.
- A continued conversation should reuse the returned `thread_id` from the
  previous response and send it as `threadId` on the next request.
- A project must be identified by either `projectId` or `projectPath`.
  Prefer `projectId` after a project has already been refined.

## GET `/api/projects`

Lists indexed projects. The frontend can call this endpoint when initializing
the project selector or restoring the last selected project.

### Characteristics

- No request body.
- Returns projects ordered by latest update time first.
- Response field names use `snake_case`.
- The returned `id` should be passed as `projectId` to RAG and vector-search
  request bodies.

### Response Body

```json
[
  {
    "id": "project:f180a56f5c2d6c1c9d064558ba21cdfa",
    "root_path": "G:\\agent\\demo",
    "name": "demo",
    "created_at": "2026-06-27T12:00:00+00:00",
    "updated_at": "2026-06-27T12:30:00+00:00"
  }
]
```

| Field | Description |
| --- | --- |
| `id` | Project id. Use this value as `projectId` in frontend requests. |
| `root_path` | Absolute project root path recorded during refinement. |
| `name` | Project display name, derived from the project directory name. |
| `created_at` | Project row creation timestamp. |
| `updated_at` | Last project refinement/update timestamp. |

## GET `/api/projects/{projectId}`

Gets one indexed project by id. Use this when the frontend already has a
stored `projectId` and needs to validate it or display project metadata.

### Path Parameters

```text
/api/projects/project:f180a56f5c2d6c1c9d064558ba21cdfa
```

| Parameter | Required | Description |
| --- | --- | --- |
| `projectId` | Yes | Project id from `GET /api/projects`, `/api/refine`, RAG responses, or vector-search responses. |

### Response Body

```json
{
  "id": "project:f180a56f5c2d6c1c9d064558ba21cdfa",
  "root_path": "G:\\agent\\demo",
  "name": "demo",
  "created_at": "2026-06-27T12:00:00+00:00",
  "updated_at": "2026-06-27T12:30:00+00:00"
}
```

If the project does not exist, the endpoint returns `404`.

## POST `/api/rag/ask/stream`

Streams a RAG answer through Server-Sent Events (SSE). This is the preferred
chat answering endpoint for the frontend because it can show progress while the
backend plans retrieval, calls tools, gathers evidence, and generates the final
answer.

### Characteristics

- Transport is `text/event-stream`.
- The connection stays open until the backend emits `event: done`.
- The backend creates the chat thread if `projectId + userId + threadId` does
  not already exist.
- If `threadId` is missing, a new `thread:<uuid-hex>` is generated and included
  in the final answer payload as `thread_id`.
- If `threadId` is provided, the backend loads recent messages for that thread
  and passes bounded conversation history to the LLM.
- The final answer is emitted as an SSE `final` event, not as the HTTP response
  body.

### Request Body

```json
{
  "question": "订单信息是在哪里存入的",
  "projectId": "project:f180a56f5c2d6c1c9d064558ba21cdfa",
  "userId": "1",
  "threadId": "thread:abc123",
  "title": "订单信息是在哪里存入的",
  "topK": 8,
  "graphDepth": 2,
  "readSource": true,
  "mode": "agentic"
}
```

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `question` | Yes | None | User question to answer. Must be a non-empty string. |
| `projectId` | Conditional | None | Refined project id. Required if `projectPath` is not provided. |
| `projectPath` | Conditional | None | Local project path. Required if `projectId` is not provided. |
| `userId` | No | `"1"` | Frontend user id. Used together with `projectId` and `threadId` to identify the chat. |
| `threadId` | No | Backend-generated `thread:<uuid-hex>` | Existing chat id. Omit it to start a new chat. |
| `title` | No | First `question` | Chat title used when the backend creates a new thread. |
| `topK` | No | `8` | Maximum number of vector/relational candidates to retrieve. Range: `1` to `50`. |
| `graphDepth` | No | `2` | Relation traversal depth. Range: `0` to `3`. |
| `readSource` | No | `true` | Whether the agent may read source snippets when needed. |
| `mode` | No | `"agentic"` | RAG execution mode. Currently only `"agentic"` is supported. |

### SSE Events

Each event follows the standard SSE shape:

```text
event: status
data: {"stage":"planning","message":"Creating retrieval plan."}

```

Known event types:

| Event | Meaning |
| --- | --- |
| `status` | Progress message such as request resolution, planning, retrieval, or answering. |
| `plan` | The planner output, including question understanding and retrieval plan. |
| `tool_result` | One retrieval tool result summary. |
| `evidence` | Aggregate evidence/tool usage summary before final answer generation. |
| `final` | Final RAG answer payload. Frontend should read `data.answer.thread_id` from this event. |
| `error` | Stream-level error. Contains `status_code` and `detail`. |
| `done` | Stream finished. The `data` object is empty. |

### Final Event Data

`final` event example:

```text
event: final
data: {"answer":{"answer":"订单信息写入 oms_order 表。","thread_id":"thread:abc123","project_id":"project:f180a56f5c2d6c1c9d064558ba21cdfa","project_path":"G:\\agent\\demo","intent":"persistence_location","rewritten_query":"订单信息是在哪里存入的","expanded_queries":["订单信息是在哪里存入的"],"used_vector_search":true,"used_relational_search":true,"used_source_reading":true,"source_reading_skipped_reason":null,"evidence":[],"source_snippets":[],"warnings":[],"mode":"agentic","used_tools":["aggregate_query"],"observations":[]}}

```

Important response fields inside `data.answer`:

| Field | Description |
| --- | --- |
| `answer` | Natural-language answer for display. |
| `thread_id` | Chat id to reuse as `threadId` in the next request. |
| `project_id` | Project id used by this answer. |
| `project_path` | Resolved project path. |
| `intent` | Planner/task intent, for example `persistence_location`. |
| `used_tools` | Retrieval tools used by the agent. |
| `evidence` | Evidence items. May be empty if no evidence was found. |
| `source_snippets` | Source snippets read by the agent. |
| `warnings` | Non-fatal warnings. |

### Frontend SSE Example

```ts
const response = await fetch("/api/rag/ask/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question,
    projectId,
    userId: userId ?? "1",
    threadId,
    title: title ?? question,
    topK: 8,
    graphDepth: 2,
    readSource: true,
    mode: "agentic",
  }),
});

const reader = response.body?.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (reader) {
  const { value, done } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const frames = buffer.split("\n\n");
  buffer = frames.pop() ?? "";

  for (const frame of frames) {
    const eventLine = frame.split("\n").find((line) => line.startsWith("event:"));
    const dataLine = frame.split("\n").find((line) => line.startsWith("data:"));
    const event = eventLine?.slice("event:".length).trim();
    const data = dataLine ? JSON.parse(dataLine.slice("data:".length)) : {};

    if (event === "final") {
      const answer = data.answer;
      // Store answer.thread_id and send it as threadId for follow-up messages.
    }
  }
}
```

## POST `/api/vector_search`

Vector-only semantic retrieval. This endpoint returns matched chunks but does
not call the LLM to generate an answer.

`/api/vector-search` is still available as a compatibility alias.

### Characteristics

- Creates a chat thread if `projectId + userId + threadId` does not already
  exist.
- If `threadId` is missing, the backend generates `thread:<uuid-hex>` and
  returns it as `thread_id`.
- If `title` is missing, the backend uses `query` as the title.
- This endpoint only creates the chat box record. It does not write user or
  assistant messages to `rag_messages`.

### Request Body

```json
{
  "query": "订单信息是在哪里存入的",
  "projectId": "project:f180a56f5c2d6c1c9d064558ba21cdfa",
  "userId": "1",
  "threadId": "thread:abc123",
  "title": "订单信息是在哪里存入的",
  "limit": 10
}
```

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `query` | Yes | None | Semantic search text. |
| `projectId` | Conditional | None | Refined project id. Required if `projectPath` is not provided. |
| `projectPath` | Conditional | None | Local project path. Required if `projectId` is not provided. |
| `userId` | No | `"1"` | Frontend user id. |
| `threadId` | No | Backend-generated `thread:<uuid-hex>` | Chat id to associate with the vector search. |
| `title` | No | First `query` | Chat title when a new chat thread is created. |
| `limit` | No | `10` | Maximum matches. Range: `1` to `100`. |

### Response Body

```json
{
  "query": "订单信息是在哪里存入的",
  "project_id": "project:f180a56f5c2d6c1c9d064558ba21cdfa",
  "user_id": "1",
  "thread_id": "thread:abc123",
  "title": "订单信息是在哪里存入的",
  "embedding_model": "text-embedding-3-large",
  "embedding_dim": 1024,
  "matches": []
}
```

Each item in `matches` contains:

| Field | Description |
| --- | --- |
| `chunk_id` | Retrieved chunk id. |
| `project_id` | Project id for the chunk. |
| `file_path` | Source file path relative to the project. |
| `title` | Chunk title. |
| `chunk_type` | Chunk type such as route, mapper method, SQL, or class. |
| `content` | Chunk content. |
| `score` | Similarity score. Higher is better. |
| `distance` | Vector distance. Lower is better. |
| `language` | Source language. |
| `start_line` / `end_line` | Source range if known. |
| `module_name` / `service_name` | Module/service metadata if available. |
| `symbol_qualified_name` | Related symbol if available. |
| `metadata` | Extra structured metadata. |

## POST `/api/rag/threads`

Creates a chat box without asking a question.

### Request Body

```json
{
  "projectId": "project:f180a56f5c2d6c1c9d064558ba21cdfa",
  "userId": "1",
  "threadId": "thread:abc123",
  "title": "订单信息是在哪里存入的",
  "firstQuestion": "订单信息是在哪里存入的"
}
```

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `projectId` | Yes | None | Project id. |
| `userId` | No | `"1"` | Frontend user id. |
| `threadId` | No | Backend-generated `thread:<uuid-hex>` | Chat id. |
| `title` | No | `firstQuestion`, then `"New chat"` | Chat title. |
| `firstQuestion` | No | None | Used as title when `title` is omitted. |

### Response Body

```json
{
  "id": 1,
  "project_id": "project:f180a56f5c2d6c1c9d064558ba21cdfa",
  "user_id": "1",
  "thread_id": "thread:abc123",
  "title": "订单信息是在哪里存入的",
  "created_at": "2026-06-27T12:00:00+00:00",
  "updated_at": "2026-06-27T12:00:00+00:00"
}
```

## GET `/api/rag/threads`

Lists chat boxes for one project and user.

### Query Parameters

```text
/api/rag/threads?projectId=project:f180a56f5c2d6c1c9d064558ba21cdfa&userId=1
```

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `projectId` | Yes | None | Project id. |
| `userId` | No | `"1"` | Frontend user id. |

Response is an array of thread objects using the same shape as
`POST /api/rag/threads`.

## GET `/api/rag/threads/{threadId}/messages`

Lists messages in one chat box.

### Query Parameters

```text
/api/rag/threads/thread:abc123/messages?projectId=project:f180a56f5c2d6c1c9d064558ba21cdfa&userId=1
```

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `threadId` | Yes | Path parameter | Chat id from the URL path. |
| `projectId` | Yes | None | Project id. |
| `userId` | No | `"1"` | Frontend user id. |

### Response Body

```json
[
  {
    "id": 10,
    "thread_id": "thread:abc123",
    "role": "user",
    "content": "订单信息是在哪里存入的",
    "created_at": "2026-06-27T12:00:00+00:00"
  },
  {
    "id": 11,
    "thread_id": "thread:abc123",
    "role": "assistant",
    "content": "订单信息写入 oms_order 表。",
    "created_at": "2026-06-27T12:00:01+00:00"
  }
]
```

## PATCH `/api/rag/threads/{threadId}/title`

Renames a chat box.

### Request Body

```json
{
  "projectId": "project:f180a56f5c2d6c1c9d064558ba21cdfa",
  "userId": "1",
  "title": "订单持久化位置"
}
```

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `threadId` | Yes | Path parameter | Chat id from the URL path. |
| `projectId` | Yes | None | Project id. |
| `userId` | No | `"1"` | Frontend user id. |
| `title` | Yes | None | New non-empty title. |

Response is the updated thread object.

## DELETE `/api/rag/threads/{threadId}`

Deletes one chat box and its messages.

### Query Parameters

```text
/api/rag/threads/thread:abc123?projectId=project:f180a56f5c2d6c1c9d064558ba21cdfa&userId=1
```

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `threadId` | Yes | Path parameter | Chat id from the URL path. |
| `projectId` | Yes | None | Project id. |
| `userId` | No | `"1"` | Frontend user id. |

### Response Body

```json
{
  "status": "deleted"
}
```

## Error Handling

Common HTTP errors:

| Status | Meaning |
| --- | --- |
| `400` | Invalid request value, for example missing both `projectId` and `projectPath`. |
| `404` | Project or chat thread was not found. |
| `422` | Request validation failed, for example missing required JSON fields. |
| `500` | Backend configuration, database, or unexpected server failure. |
| `502` | LLM invocation failed. |

For `/api/rag/ask/stream`, errors after the SSE connection starts are emitted
as:

```text
event: error
data: {"status_code":400,"detail":"project_id or project_path is required."}

```
