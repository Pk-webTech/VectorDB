package com.vectordb.models;

import java.util.List;

public class DocItem {
    public int id;
    public String title;
    public String text;
    public List<Float> emb;

    public DocItem() {
    }

    public DocItem(int id, String title, String text, List<Float> emb) {
        this.id = id;
        this.title = title;
        this.text = text;
        this.emb = emb;
    }
}