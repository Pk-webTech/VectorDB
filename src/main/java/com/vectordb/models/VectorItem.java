package com.vectordb.models;

import java.util.List;
import java.util.ArrayList;

public class VectorItem {
    public int id;
    public String metadata;
    public String category;
    public List<Float> emb;

    public VectorItem() {}

    public VectorItem(int id, String metadata, String category, List<Float> emb) {
        this.id = id;
        this.metadata = metadata;
        this.category = category;
        this.emb = new ArrayList<>(emb);
    }
}