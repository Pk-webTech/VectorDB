package com.vectordb.algorithms;

import com.vectordb.models.VectorItem;

import java.util.*;
import java.util.function.BiFunction;

public class KDTree {
    private static class Node {
        VectorItem item;
        Node left, right;

        Node(VectorItem item) {
            this.item = item;
        }
    }

    private Node root;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    public synchronized void insert(VectorItem v) {
        root = insert(root, v, 0);
    }

    private Node insert(Node n, VectorItem v, int depth) {
        if (n == null)
            return new Node(v);
        int axis = depth % dims;
        if (v.emb.get(axis) < n.item.emb.get(axis))
            n.left = insert(n.left, v, depth + 1);
        else
            n.right = insert(n.right, v, depth + 1);
        return n;
    }

    public synchronized List<float[]> knn(List<Float> q, int k,
            BiFunction<List<Float>, List<Float>, Float> dist) {
        // max-heap: (distance, id)
        PriorityQueue<float[]> heap = new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));
        knn(root, q, k, 0, dist, heap);
        List<float[]> results = new ArrayList<>(heap);
        results.sort(Comparator.comparingDouble(r -> r[0]));
        return results;
    }

    private void knn(Node n, List<Float> q, int k, int depth,
            BiFunction<List<Float>, List<Float>, Float> dist,
            PriorityQueue<float[]> heap) {
        if (n == null)
            return;
        float dn = dist.apply(q, n.item.emb);
        if (heap.size() < k || dn < heap.peek()[0]) {
            heap.offer(new float[] { dn, n.item.id });
            if (heap.size() > k)
                heap.poll();
        }
        int axis = depth % dims;
        float diff = q.get(axis) - n.item.emb.get(axis);
        Node closer = diff < 0 ? n.left : n.right;
        Node farther = diff < 0 ? n.right : n.left;
        knn(closer, q, k, depth + 1, dist, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek()[0])
            knn(farther, q, k, depth + 1, dist, heap);
    }

    public synchronized void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items)
            root = insert(root, v, 0);
    }
}