package com.vectordb.util;

import java.net.*;
import java.net.http.*;
import java.time.Duration;
import java.util.*;

/**
 * Java HTTP client wrapping the local Ollama REST API.
 * Mirrors the C++ OllamaClient exactly.
 *
 * Install Ollama: https://ollama.com
 * Then run:
 * ollama pull nomic-embed-text
 * ollama pull llama3.2
 */
public class OllamaClient {

    public String embedModel = "nomic-embed-text";
    public String genModel = "llama3.2";

    private final String baseUrl;
    private final HttpClient client;

    public OllamaClient() {
        this("127.0.0.1", 11434);
    }

    public OllamaClient(String host, int port) {
        this.baseUrl = "http://" + host + ":" + port;
        this.client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .build();
    }

    // ── Availability check ─────────────────────────────────────────────────
    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/tags"))
                    .timeout(Duration.ofSeconds(2))
                    .GET().build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            return res.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    // ── Text embedding ─────────────────────────────────────────────────────
    public List<Float> embed(String text) {
        try {
            String body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/embeddings"))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200)
                return Collections.emptyList();
            return parseEmbedding(res.body());
        } catch (Exception e) {
            return Collections.emptyList();
        }
    }

    // ── Text generation ────────────────────────────────────────────────────
    public String generate(String prompt) {
        try {
            String body = "{\"model\":\"" + genModel + "\",\"prompt\":\"" + esc(prompt) + "\",\"stream\":false}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/generate"))
                    .timeout(Duration.ofSeconds(180))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200)
                return "ERROR: Ollama unavailable. Run: ollama serve";
            return parseResponse(res.body());
        } catch (Exception e) {
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
    }

    // ── JSON helpers ───────────────────────────────────────────────────────
    private String esc(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private List<Float> parseEmbedding(String body) {
        int p = body.indexOf("\"embeddings\"");
        if (p < 0) {
            p = body.indexOf("\"embedding\"");
        }
        if (p < 0)
            return Collections.emptyList();

        int start = body.indexOf('[', p);
        if (start < 0)
            return Collections.emptyList();

        // Find matching ]
        int end = start + 1, depth = 1;
        while (end < body.length() && depth > 0) {
            char c = body.charAt(end);
            if (c == '[')
                depth++;
            else if (c == ']')
                depth--;
            end++;
        }
        String inner = body.substring(start + 1, end - 1).trim();
        List<Float> result = new ArrayList<>();
        for (String tok : inner.split(",")) {
            tok = tok.trim();
            if (!tok.isEmpty()) {
                try {
                    result.add(Float.parseFloat(tok));
                } catch (NumberFormatException ignored) {
                }
            }
        }
        return result;
    }

    private String parseResponse(String body) {
        // extract "response":"..."
        int p = body.indexOf("\"response\"");
        if (p < 0)
            return "";
        p = body.indexOf('"', body.indexOf(':', p)) + 1;
        if (p <= 0)
            return "";
        StringBuilder sb = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"')
                break;
            if (c == '\\' && p + 1 < body.length()) {
                char nc = body.charAt(p + 1);
                switch (nc) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    default -> sb.append(nc);
                }
                p += 2;
                continue;
            }
            sb.append(c);
            p++;
        }
        return sb.toString();
    }
}