# Policy RAG

A Python assistant for asking questions about a fictional homeowners insurance policy. The corpus separates coverage rules from financial limits. For example, the sewer backup endorsement defines when coverage applies, while its payment schedule defines the deductible and limit. The assistant retrieves both clauses and can use a local LLM to answer with citations.

The policy and benchmark are fictional and were written for this demo.

## Run locally

Use Python 3.10 or newer. The offline mode uses the standard library.

```sh
cd policy-rag
python3 -m unittest -v
python3 evaluate.py
python3 app.py
```

In another terminal, ask a question with graph expansion enabled.

```sh
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Does the sewer backup endorsement cover my dwelling, and what is the deductible?","use_graph":true}'
```

Offline mode returns the matching clauses as `evidence_only`. Repeat with `"use_graph":false` to see which clause the lexical search misses. `GET /health` reports whether the API is listening.

## Enable local embeddings and generated answers

Start the Ollama app or run `ollama serve` in a separate terminal. Stop the offline API before restarting it with the models enabled.

```sh
ollama pull nomic-embed-text
ollama pull llama3.1
EMBEDDING_MODEL=nomic-embed-text CHAT_MODEL=llama3.1 python3 app.py
```

Repeat the same request. The API returns the cited policy text alongside the generated answer. In a local Llama 3.1 smoke test, the answer cited both correct clauses and gave the $500 deductible, but it omitted the declarations requirement. Use the returned sources to spot that gap. The same smoke test declined to answer a dental surgery question.

## How it works

1. **Embed.** Index one clause per document with Ollama embeddings or a sparse IDF-weighted word vector. The offline vector uses binary term frequency and downweights common terms.
2. **Retrieve.** Rank the in-memory index by cosine similarity and apply a minimum score. This is an exhaustive search across the small corpus.
3. **Follow the policy graph.** Represent directed clause dependencies as an adjacency list. In one hop, put the best match’s required clauses directly after it and deduplicate before applying the result limit.
4. **Generate.** Send the question and retrieved clauses to the model with a structured answer contract. Return only citations to clauses provided to the model. Citation membership alone cannot verify that a clause supports every claim.

## Evaluate

Run `python3 evaluate.py` for the offline comparison or `EMBEDDING_MODEL=nomic-embed-text python3 evaluate.py` for the dense embedding comparison. Both runs use the same cutoff and benchmark labels with and without the graph. Recall measures how much of the expected evidence was found, MRR measures the position of the first relevant clause, and the all-sources rate measures whether every required clause was found.

On this 10-question benchmark, both embedding modes went from **78.6% to 100% recall@2** and from **4/7 to 7/7 complete evidence sets** when the graph was enabled. The offline retriever rejected 1 of 3 out-of-scope questions and the dense retriever rejected 0 of 3 at the demo thresholds. This benchmark measures retrieval, including whether it passes any context to the model. It does not measure answer correctness or generalization to other policies.

For an interview demo, use the sewer backup question, show the graph edge in `policies.json`, and explain why the baseline found the coverage rule but missed the schedule. Then show an out-of-scope question to discuss the difference between retrieval confidence, a model declining to answer, and a verified answer.
