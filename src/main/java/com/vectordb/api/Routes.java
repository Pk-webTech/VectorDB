package com.vectordb.api;

import com.vectordb.algorithms.HNSW;
import com.vectordb.db.DocumentDB;
import com.vectordb.db.VectorDB;
import com.vectordb.models.DocItem;
import com.vectordb.models.VectorItem;
import com.vectordb.util.*;
import io.javalin.Javalin;
import io.javalin.http.Context;

import java.util.List;

import static com.vectordb.util.JsonUtil.*;

public class Routes {

    private static final int DIMS = 16;

    public static void register(Javalin app, VectorDB db, DocumentDB docDB, OllamaClient ollama) {

        // ── CORS preflight ─────────────────────────────────────────────────
        app.before(ctx -> {
            ctx.header("Access-Control-Allow-Origin", "*");
            ctx.header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
            ctx.header("Access-Control-Allow-Headers", "Content-Type");
        });
        app.options("/*", ctx -> ctx.status(204));

        // ── GET / — serve index.html ───────────────────────────────────────
        app.get("/", ctx -> {
            var stream = Routes.class.getClassLoader().getResourceAsStream("index.html");
            if (stream == null) {
                ctx.status(404).result("index.html not found");
                return;
            }
            ctx.contentType("text/html").result(stream);
        });

        // ══════════════════════════════════════════════════════════════════
        // DEMO VECTOR ENDPOINTS
        // ══════════════════════════════════════════════════════════════════

        // GET /search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw
        app.get("/search", ctx -> {
            List<Float> q = parseVec(ctx.queryParam("v"));
            if (q.size() != DIMS) {
                ctx.contentType("application/json")
                        .result("{\"error\":\"need " + DIMS + "D vector\"}");
                return;
            }
            int k = intParam(ctx, "k", 5);
            String metric = ctx.queryParamAsClass("metric", String.class).getOrDefault("cosine");
            String algo = ctx.queryParamAsClass("algo", String.class).getOrDefault("hnsw");

            var out = db.search(q, k, metric, algo);
            var sb = new StringBuilder("{\"results\":[");
            for (int i = 0; i < out.hits.size(); i++) {
                if (i > 0)
                    sb.append(',');
                var h = out.hits.get(i);
                sb.append("{\"id\":").append(h.id)
                        .append(",\"metadata\":").append(jS(h.meta))
                        .append(",\"category\":").append(jS(h.cat))
                        .append(",\"distance\":").append(String.format("%.6f", h.dist))
                        .append(",\"embedding\":").append(jVec(h.emb))
                        .append('}');
            }
            sb.append("],\"latencyUs\":").append(out.latencyUs)
                    .append(",\"algo\":").append(jS(out.algo))
                    .append(",\"metric\":").append(jS(out.metric))
                    .append('}');
            ctx.contentType("application/json").result(sb.toString());
        });

        // POST /insert {"metadata":"..","category":"..","embedding":[..]}
        app.post("/insert", ctx -> {
            String body = ctx.body();
            String meta = extractStr(body, "metadata");
            String cat = extractStr(body, "category");
            List<Float> emb = extractEmbedding(body);
            if (meta.isEmpty() || emb.size() != DIMS) {
                ctx.contentType("application/json").result("{\"error\":\"invalid body\"}");
                return;
            }
            int id = db.insert(meta, cat, emb, com.vectordb.util.DistanceMetrics.get("cosine"));
            ctx.contentType("application/json").result("{\"id\":" + id + "}");
        });

        // DELETE /delete/:id
        app.delete("/delete/{id}", ctx -> {
            int id = Integer.parseInt(ctx.pathParam("id"));
            boolean ok = db.remove(id);
            ctx.contentType("application/json")
                    .result("{\"ok\":" + ok + "}");
        });

        // GET /items
        app.get("/items", ctx -> {
            List<VectorItem> items = db.all();
            var sb = new StringBuilder("[");
            for (int i = 0; i < items.size(); i++) {
                if (i > 0)
                    sb.append(',');
                var v = items.get(i);
                sb.append("{\"id\":").append(v.id)
                        .append(",\"metadata\":").append(jS(v.metadata))
                        .append(",\"category\":").append(jS(v.category))
                        .append(",\"embedding\":").append(jVec(v.emb))
                        .append('}');
            }
            sb.append(']');
            ctx.contentType("application/json").result(sb.toString());
        });

        // GET /benchmark?v=...&k=5&metric=cosine
        app.get("/benchmark", ctx -> {
            List<Float> q = parseVec(ctx.queryParam("v"));
            if (q.size() != DIMS) {
                ctx.contentType("application/json")
                        .result("{\"error\":\"need " + DIMS + "D vector\"}");
                return;
            }
            int k = intParam(ctx, "k", 5);
            String metric = ctx.queryParamAsClass("metric", String.class).getOrDefault("cosine");
            var b = db.benchmark(q, k, metric);
            ctx.contentType("application/json").result(
                    "{\"bruteforceUs\":" + b.bfUs +
                            ",\"kdtreeUs\":" + b.kdUs +
                            ",\"hnswUs\":" + b.hnswUs +
                            ",\"itemCount\":" + b.itemCount + "}");
        });

        // GET /hnsw-info
        app.get("/hnsw-info", ctx -> {
            HNSW.GraphInfo gi = db.hnswInfo();
            var sb = new StringBuilder();
            sb.append("{\"topLayer\":").append(gi.topLayer)
                    .append(",\"nodeCount\":").append(gi.nodeCount)
                    .append(",\"nodesPerLayer\":[");
            for (int i = 0; i < gi.nodesPerLayer.length; i++) {
                if (i > 0)
                    sb.append(',');
                sb.append(gi.nodesPerLayer[i]);
            }
            sb.append("],\"edgesPerLayer\":[");
            for (int i = 0; i < gi.edgesPerLayer.length; i++) {
                if (i > 0)
                    sb.append(',');
                sb.append(gi.edgesPerLayer[i]);
            }
            sb.append("],\"nodes\":[");
            for (int i = 0; i < gi.nodes.size(); i++) {
                if (i > 0)
                    sb.append(',');
                var n = gi.nodes.get(i);
                sb.append("{\"id\":").append(n.id)
                        .append(",\"metadata\":").append(jS(n.metadata))
                        .append(",\"category\":").append(jS(n.category))
                        .append(",\"maxLyr\":").append(n.maxLayer).append('}');
            }
            sb.append("],\"edges\":[");
            for (int i = 0; i < gi.edges.size(); i++) {
                if (i > 0)
                    sb.append(',');
                var e = gi.edges.get(i);
                sb.append("{\"src\":").append(e.src)
                        .append(",\"dst\":").append(e.dst)
                        .append(",\"lyr\":").append(e.layer).append('}');
            }
            sb.append("]}");
            ctx.contentType("application/json").result(sb.toString());
        });

        // GET /stats
        app.get("/stats", ctx -> ctx.contentType("application/json").result(
                "{\"count\":" + db.size() +
                        ",\"dims\":" + DIMS +
                        ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]" +
                        ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}"));

        // ══════════════════════════════════════════════════════════════════
        // DOCUMENT + RAG ENDPOINTS
        // ══════════════════════════════════════════════════════════════════

        // POST /doc/insert {"title":"..","text":".."}
        app.post("/doc/insert", ctx -> {
            String body = ctx.body();
            String title = extractStr(body, "title");
            String text = extractStr(body, "text");
            if (title.isEmpty() || text.isEmpty()) {
                ctx.contentType("application/json")
                        .result("{\"error\":\"need title and text\"}");
                return;
            }

            List<String> chunks = TextChunker.chunk(text, 250, 30);
            var ids = new StringBuilder();
            int inserted = 0;

            for (int i = 0; i < chunks.size(); i++) {
                List<Float> emb = ollama.embed(chunks.get(i));
                if (emb.isEmpty()) {
                    ctx.contentType("application/json").result(
                            "{\"error\":\"Ollama unavailable. " +
                                    "Install from https://ollama.com then run: " +
                                    "ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                    return;
                }
                String chunkTitle = (chunks.size() > 1)
                        ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                        : title;
                int id = docDB.insert(chunkTitle, chunks.get(i), emb);
                if (inserted > 0)
                    ids.append(',');
                ids.append(id);
                inserted++;
            }

            ctx.contentType("application/json").result(
                    "{\"ids\":[" + ids + "]" +
                            ",\"chunks\":" + chunks.size() +
                            ",\"dims\":" + docDB.getDims() + "}");
        });

        // DELETE /doc/delete/:id
        app.delete("/doc/delete/{id}", ctx -> {
            int id = Integer.parseInt(ctx.pathParam("id"));
            boolean ok = docDB.remove(id);
            ctx.contentType("application/json").result("{\"ok\":" + ok + "}");
        });

        // GET /doc/list
        app.get("/doc/list", ctx -> {
            List<DocItem> docs = docDB.all();
            var sb = new StringBuilder("[");
            for (int i = 0; i < docs.size(); i++) {
                if (i > 0)
                    sb.append(',');
                var d = docs.get(i);
                String preview = d.text.length() > 120
                        ? d.text.substring(0, 120) + "…"
                        : d.text;
                int wordCount = d.text.split("\\s+").length;
                sb.append("{\"id\":").append(d.id)
                        .append(",\"title\":").append(jS(d.title))
                        .append(",\"preview\":").append(jS(preview))
                        .append(",\"words\":").append(wordCount)
                        .append('}');
            }
            sb.append(']');
            ctx.contentType("application/json").result(sb.toString());
        });

        // POST /doc/search {"question":"..","k":3} — fast retrieval for UI
        app.post("/doc/search", ctx -> {
            String body = ctx.body();
            String question = extractStr(body, "question");
            int k = extractInt(body, "k", 3);
            if (question.isEmpty()) {
                ctx.contentType("application/json").result("{\"error\":\"need question\"}");
                return;
            }
            List<Float> qEmb = ollama.embed(question);
            if (qEmb.isEmpty()) {
                ctx.contentType("application/json").result("{\"error\":\"Ollama unavailable\"}");
                return;
            }
            var hits = docDB.search(qEmb, k);
            var sb = new StringBuilder("{\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0)
                    sb.append(',');
                var h = hits.get(i);
                sb.append("{\"id\":").append(h.doc.id)
                        .append(",\"title\":").append(jS(h.doc.title))
                        .append(",\"distance\":").append(String.format("%.4f", h.dist))
                        .append('}');
            }
            sb.append("]}");
            ctx.contentType("application/json").result(sb.toString());
        });

        // POST /doc/ask {"question":"..","k":3} — full RAG pipeline
        app.post("/doc/ask", ctx -> {
            String body = ctx.body();
            String question = extractStr(body, "question");
            int k = extractInt(body, "k", 3);
            if (question.isEmpty()) {
                ctx.contentType("application/json").result("{\"error\":\"need question\"}");
                return;
            }

            // Step 1: embed question
            List<Float> qEmb = ollama.embed(question);
            if (qEmb.isEmpty()) {
                ctx.contentType("application/json").result("{\"error\":\"Ollama unavailable\"}");
                return;
            }

            // Step 2: retrieve top-k chunks
            var hits = docDB.search(qEmb, k);

            // Step 3: build prompt
            var ctx_sb = new StringBuilder();
            for (int i = 0; i < hits.size(); i++) {
                ctx_sb.append('[').append(i + 1).append("] ")
                        .append(hits.get(i).doc.title).append(":\n")
                        .append(hits.get(i).doc.text).append("\n\n");
            }
            String prompt = "You are a helpful assistant. Answer the user's question directly. " +
                    "Use the provided context if it contains relevant information. " +
                    "If it doesn't, just use your own general knowledge. " +
                    "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like " +
                    "'the context doesn't mention'. Just answer the question naturally.\n\n" +
                    "Context:\n" + ctx_sb +
                    "Question: " + question + "\n\nAnswer:";

            // Step 4: generate answer
            String answer = ollama.generate(prompt);

            // Step 5: return everything
            var sb = new StringBuilder();
            sb.append("{\"answer\":").append(jS(answer))
                    .append(",\"model\":").append(jS(ollama.genModel))
                    .append(",\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0)
                    sb.append(',');
                var h = hits.get(i);
                sb.append("{\"id\":").append(h.doc.id)
                        .append(",\"title\":").append(jS(h.doc.title))
                        .append(",\"text\":").append(jS(h.doc.text))
                        .append(",\"distance\":").append(String.format("%.4f", h.dist))
                        .append('}');
            }
            sb.append("],\"docCount\":").append(docDB.size()).append('}');
            ctx.contentType("application/json").result(sb.toString());
        });

        // GET /status
        app.get("/status", ctx -> {
            boolean up = ollama.isAvailable();
            ctx.contentType("application/json").result(
                    "{\"ollamaAvailable\":" + up +
                            ",\"embedModel\":" + jS(ollama.embedModel) +
                            ",\"genModel\":" + jS(ollama.genModel) +
                            ",\"docCount\":" + docDB.size() +
                            ",\"docDims\":" + docDB.getDims() +
                            ",\"demoDims\":" + DIMS +
                            ",\"demoCount\":" + db.size() + "}");
        });
    }

    // ── Helper ─────────────────────────────────────────────────────────────
    private static int intParam(Context ctx, String name, int def) {
        String v = ctx.queryParam(name);
        if (v == null)
            return def;
        try {
            return Integer.parseInt(v);
        } catch (NumberFormatException e) {
            return def;
        }
    }
}
