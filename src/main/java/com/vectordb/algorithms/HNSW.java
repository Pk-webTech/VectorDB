package com.vectordb.algorithms;

import com.vectordb.models.VectorItem;

import java.util.*;
import java.util.function.BiFunction;

/**
 * Hierarchical Navigable Small World (HNSW) — Java port of the C++
 * implementation.
 * Same algorithm used by Pinecone, Weaviate, Chroma, Milvus.
 */
public class HNSW {

    // ── Graph node ──────────────────────────────────────────────────────
    private static class Node {
        VectorItem item;
        int maxLayer;
        List<List<Integer>> neighbors; // neighbors[layer] = list of node IDs

        Node(VectorItem item, int maxLayer) {
            this.item = item;
            this.maxLayer = maxLayer;
            this.neighbors = new ArrayList<>();
            for (int i = 0; i <= maxLayer; i++)
                neighbors.add(new ArrayList<>());
        }
    }

    // ── Graph Info structures (for /hnsw-info endpoint) ─────────────────
    public static class NodeView {
        public int id;
        public String metadata, category;
        public int maxLayer;

        public NodeView(int id, String metadata, String category, int maxLayer) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.maxLayer = maxLayer;
        }
    }

    public static class EdgeView {
        public int src, dst, layer;

        public EdgeView(int src, int dst, int layer) {
            this.src = src;
            this.dst = dst;
            this.layer = layer;
        }
    }

    public static class GraphInfo {
        public int topLayer, nodeCount;
        public int[] nodesPerLayer, edgesPerLayer;
        public List<NodeView> nodes;
        public List<EdgeView> edges;
    }

    // ── Fields ───────────────────────────────────────────────────────────
    private final Map<Integer, Node> graph = new HashMap<>();
    private final int M, M0, efBuild;
    private final double mL;
    private int topLayer = -1;
    private int entryPoint = -1;
    private final Random rng;

    public HNSW(int m, int efBuild) {
        this.M = m;
        this.M0 = 2 * m;
        this.efBuild = efBuild;
        this.mL = 1.0 / Math.log(m);
        this.rng = new Random(42);
    }

    private int randomLevel() {
        return (int) Math.floor(-Math.log(rng.nextDouble()) * mL);
    }

    // ── Layer search (beam search) ────────────────────────────────────────
    private List<float[]> searchLayer(List<Float> q, int ep, int ef, int layer,
            BiFunction<List<Float>, List<Float>, Float> dist) {
        Set<Integer> visited = new HashSet<>();
        // min-heap: (dist, id) → candidates to explore
        PriorityQueue<float[]> candidates = new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
        // max-heap: (dist, id) → found results
        PriorityQueue<float[]> found = new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));

        float d0 = dist.apply(q, graph.get(ep).item.emb);
        visited.add(ep);
        candidates.offer(new float[] { d0, ep });
        found.offer(new float[] { d0, ep });

        while (!candidates.isEmpty()) {
            float[] c = candidates.poll();
            float cd = c[0];
            int cid = (int) c[1];
            if (found.size() >= ef && cd > found.peek()[0])
                break;

            Node cn = graph.get(cid);
            if (cn == null || layer >= cn.neighbors.size())
                continue;

            for (int nid : cn.neighbors.get(layer)) {
                if (visited.contains(nid) || !graph.containsKey(nid))
                    continue;
                visited.add(nid);
                float nd = dist.apply(q, graph.get(nid).item.emb);
                if (found.size() < ef || nd < found.peek()[0]) {
                    candidates.offer(new float[] { nd, nid });
                    found.offer(new float[] { nd, nid });
                    if (found.size() > ef)
                        found.poll();
                }
            }
        }

        List<float[]> result = new ArrayList<>(found);
        result.sort(Comparator.comparingDouble(a -> a[0]));
        return result;
    }

    // ── Select M nearest neighbors ─────────────────────────────────────────
    private List<Integer> selectNeighbors(List<float[]> candidates, int maxM) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < Math.min(maxM, candidates.size()); i++) {
            result.add((int) candidates.get(i)[1]);
        }
        return result;
    }

    // ── Insert ───────────────────────────────────────────────────────────
    public synchronized void insert(VectorItem item,
            BiFunction<List<Float>, List<Float>, Float> dist) {
        int id = item.id;
        int lvl = randomLevel();
        graph.put(id, new Node(item, lvl));

        if (entryPoint == -1) {
            entryPoint = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPoint;

        // Traverse layers above lvl — greedy descent to find good entry
        for (int lc = topLayer; lc > lvl; lc--) {
            Node epNode = graph.get(ep);
            if (epNode != null && lc < epNode.neighbors.size()) {
                List<float[]> W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.isEmpty())
                    ep = (int) W.get(0)[1];
            }
        }

        // Insert from min(topLayer, lvl) down to 0
        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<float[]> W = searchLayer(item.emb, ep, efBuild, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> selected = selectNeighbors(W, maxM);

            Node newNode = graph.get(id);
            while (newNode.neighbors.size() <= lc)
                newNode.neighbors.add(new ArrayList<>());
            newNode.neighbors.set(lc, new ArrayList<>(selected));

            // Add back-edges, pruning if needed
            for (int nid : selected) {
                Node neighbor = graph.get(nid);
                if (neighbor == null)
                    continue;
                while (neighbor.neighbors.size() <= lc)
                    neighbor.neighbors.add(new ArrayList<>());
                List<Integer> conn = neighbor.neighbors.get(lc);
                conn.add(id);
                if (conn.size() > maxM) {
                    // Prune: keep only the M nearest
                    List<float[]> ds = new ArrayList<>();
                    for (int c : conn) {
                        Node cn = graph.get(c);
                        if (cn != null) {
                            float d = dist.apply(neighbor.item.emb, cn.item.emb);
                            ds.add(new float[] { d, c });
                        }
                    }
                    ds.sort(Comparator.comparingDouble(a -> a[0]));
                    conn.clear();
                    for (int i = 0; i < Math.min(maxM, ds.size()); i++) {
                        conn.add((int) ds.get(i)[1]);
                    }
                }
            }
            if (!W.isEmpty())
                ep = (int) W.get(0)[1];
        }

        if (lvl > topLayer) {
            topLayer = lvl;
            entryPoint = id;
        }
    }

    // ── Search (KNN) ─────────────────────────────────────────────────────
    public synchronized List<float[]> knn(List<Float> q, int k, int ef,
            BiFunction<List<Float>, List<Float>, Float> dist) {
        if (entryPoint == -1)
            return Collections.emptyList();
        int ep = entryPoint;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = graph.get(ep);
            if (epNode != null && lc < epNode.neighbors.size()) {
                List<float[]> W = searchLayer(q, ep, 1, lc, dist);
                if (!W.isEmpty())
                    ep = (int) W.get(0)[1];
            }
        }
        List<float[]> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        return W.subList(0, Math.min(k, W.size()));
    }

    // ── Remove ───────────────────────────────────────────────────────────
    public synchronized void remove(int id) {
        if (!graph.containsKey(id))
            return;
        for (Node nd : graph.values()) {
            for (List<Integer> layer : nd.neighbors) {
                layer.removeIf(n -> n == id);
            }
        }
        if (entryPoint == id) {
            entryPoint = -1;
            for (int nid : graph.keySet()) {
                if (nid != id) {
                    entryPoint = nid;
                    break;
                }
            }
        }
        graph.remove(id);
    }

    // ── Graph info (for /hnsw-info endpoint) ─────────────────────────────
    public synchronized GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer = topLayer;
        gi.nodeCount = graph.size();
        int maxL = Math.max(topLayer + 1, 1);
        gi.nodesPerLayer = new int[maxL];
        gi.edgesPerLayer = new int[maxL];
        gi.nodes = new ArrayList<>();
        gi.edges = new ArrayList<>();

        for (Map.Entry<Integer, Node> e : graph.entrySet()) {
            int id = e.getKey();
            Node nd = e.getValue();
            gi.nodes.add(new NodeView(id, nd.item.metadata, nd.item.category, nd.maxLayer));
            for (int lc = 0; lc <= nd.maxLayer && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < nd.neighbors.size()) {
                    for (int nid : nd.neighbors.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.add(new EdgeView(id, nid, lc));
                        }
                    }
                }
            }
        }
        return gi;
    }

    public synchronized int size() {
        return graph.size();
    }
}