package com.vectordb.algorithms;

import com.vectordb.models.VectorItem;

import java.util.*;
import java.util.function.BiFunction;

public class BruteForce {
    private final List<VectorItem> items = new ArrayList<>();

    public synchronized void insert(VectorItem v) {
        items.add(new VectorItem(v.id, v.metadata, v.category, v.emb));
    }

    public synchronized List<float[]> knn(List<Float> q, int k,
            BiFunction<List<Float>, List<Float>, Float> dist) {
        List<float[]> results = new ArrayList<>();
        for (VectorItem v : items) {
            results.add(new float[] { dist.apply(q, v.emb), v.id });
        }
        results.sort(Comparator.comparingDouble(r -> r[0]));
        return results.subList(0, Math.min(k, results.size()));
    }

    public synchronized void remove(int id) {
        items.removeIf(v -> v.id == id);
    }

    public synchronized List<VectorItem> getItems() {
        return new ArrayList<>(items);
    }

    public synchronized int size() {
        return items.size();
    }
}