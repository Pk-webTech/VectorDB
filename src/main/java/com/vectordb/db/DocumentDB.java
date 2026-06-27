package com.vectordb.db;

import com.vectordb.algorithms.BruteForce;
import com.vectordb.algorithms.HNSW;
import com.vectordb.models.DocItem;
import com.vectordb.models.VectorItem;
import com.vectordb.util.DistanceMetrics;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Document store backed by HNSW over real 768-dimensional Ollama embeddings.
 */
public class DocumentDB {

    private final Map<Integer, DocItem> store = new LinkedHashMap<>();
    private final HNSW hnsw = new HNSW(16, 200);
    private final BruteForce bf = new BruteForce();
    private final AtomicInteger nextId = new AtomicInteger(1);
    private int dims = 0;

    public synchronized int insert(String title, String text, List<Float> emb) {
        if (dims == 0)
            dims = emb.size();
        DocItem item = new DocItem(nextId.getAndIncrement(), title, text, emb);
        store.put(item.id, item);
        VectorItem vi = new VectorItem(item.id, title, "doc", emb);
        hnsw.insert(vi, DistanceMetrics::cosine);
        bf.insert(vi);
        return item.id;
    }

    public static class DocHit {
        public float dist;
        public DocItem doc;

        public DocHit(float dist, DocItem doc) {
            this.dist = dist;
            this.doc = doc;
        }
    }

    public synchronized List<DocHit> search(List<Float> q, int k) {
        if (store.isEmpty())
            return Collections.emptyList();
        List<float[]> raw = (store.size() < 10)
                ? bf.knn(q, k, DistanceMetrics::cosine)
                : hnsw.knn(q, k, 50, DistanceMetrics::cosine);

        List<DocHit> out = new ArrayList<>();
        for (float[] r : raw) {
            int id = (int) r[1];
            DocItem doc = store.get(id);
            if (doc != null && r[0] <= 0.7f)
                out.add(new DocHit(r[0], doc));
        }
        return out;
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id))
            return false;
        store.remove(id);
        hnsw.remove(id);
        bf.remove(id);
        return true;
    }

    public synchronized List<DocItem> all() {
        return new ArrayList<>(store.values());
    }

    public synchronized int size() {
        return store.size();
    }

    public synchronized int getDims() {
        return dims;
    }
}