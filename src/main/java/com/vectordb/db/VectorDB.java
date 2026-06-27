package com.vectordb.db;

import com.vectordb.algorithms.BruteForce;
import com.vectordb.algorithms.HNSW;
import com.vectordb.algorithms.KDTree;
import com.vectordb.models.VectorItem;
import com.vectordb.util.DistanceMetrics;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;

/**
 * Unified Vector Database wrapping BruteForce, KD-Tree and HNSW.
 * Used for the 16-dimensional demo vectors.
 */
public class VectorDB {

    public static class Hit {
        public int id;
        public String meta, cat;
        public List<Float> emb;
        public float dist;

        public Hit(int id, String meta, String cat, List<Float> emb, float dist) {
            this.id = id;
            this.meta = meta;
            this.cat = cat;
            this.emb = emb;
            this.dist = dist;
        }
    }

    public static class SearchResult {
        public List<Hit> hits;
        public long latencyUs;
        public String algo, metric;

        public SearchResult(List<Hit> hits, long us, String algo, String metric) {
            this.hits = hits;
            this.latencyUs = us;
            this.algo = algo;
            this.metric = metric;
        }
    }

    public static class BenchResult {
        public long bfUs, kdUs, hnswUs;
        public int itemCount;
    }

    // ── State ─────────────────────────────────────────────────────────────
    private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw;
    private final AtomicInteger nextId = new AtomicInteger(1);
    public final int dims;

    public VectorDB(int dims) {
        this.dims = dims;
        this.kdt = new KDTree(dims);
        this.hnsw = new HNSW(16, 200);
    }

    // ── Insert ────────────────────────────────────────────────────────────
    public synchronized int insert(String meta, String cat, List<Float> emb,
            BiFunction<List<Float>, List<Float>, Float> dist) {
        VectorItem v = new VectorItem(nextId.getAndIncrement(), meta, cat, emb);
        store.put(v.id, v);
        bf.insert(v);
        kdt.insert(v);
        hnsw.insert(v, dist);
        return v.id;
    }

    // ── Remove ────────────────────────────────────────────────────────────
    public synchronized boolean remove(int id) {
        if (!store.containsKey(id))
            return false;
        store.remove(id);
        bf.remove(id);
        hnsw.remove(id);
        kdt.rebuild(new ArrayList<>(store.values()));
        return true;
    }

    // ── Search ────────────────────────────────────────────────────────────
    public synchronized SearchResult search(List<Float> q, int k, String metric, String algo) {
        BiFunction<List<Float>, List<Float>, Float> dist = DistanceMetrics.get(metric);
        long t0 = System.nanoTime();

        List<float[]> raw;
        raw = switch (algo == null ? "" : algo.toLowerCase()) {
            case "bruteforce" -> bf.knn(q, k, dist);
            case "kdtree" -> kdt.knn(q, k, dist);
            default -> hnsw.knn(q, k, 50, dist);
        };

        long us = (System.nanoTime() - t0) / 1_000;

        List<Hit> hits = new ArrayList<>();
        for (float[] r : raw) {
            int id = (int) r[1];
            VectorItem v = store.get(id);
            if (v != null)
                hits.add(new Hit(id, v.metadata, v.category, v.emb, r[0]));
        }
        return new SearchResult(hits, us, algo, metric);
    }

    // ── Benchmark ─────────────────────────────────────────────────────────
    public synchronized BenchResult benchmark(List<Float> q, int k, String metric) {
        BiFunction<List<Float>, List<Float>, Float> dist = DistanceMetrics.get(metric);
        BenchResult b = new BenchResult();
        b.itemCount = store.size();

        long t;
        t = System.nanoTime();
        bf.knn(q, k, dist);
        b.bfUs = (System.nanoTime() - t) / 1_000;
        t = System.nanoTime();
        kdt.knn(q, k, dist);
        b.kdUs = (System.nanoTime() - t) / 1_000;
        t = System.nanoTime();
        hnsw.knn(q, k, 50, dist);
        b.hnswUs = (System.nanoTime() - t) / 1_000;
        return b;
    }

    // ── Accessors ─────────────────────────────────────────────────────────
    public synchronized List<VectorItem> all() {
        return new ArrayList<>(store.values());
    }

    public synchronized int size() {
        return store.size();
    }

    public HNSW.GraphInfo hnswInfo() {
        return hnsw.getInfo();
    }
}