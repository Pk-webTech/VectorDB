package com.vectordb;

import com.vectordb.api.Routes;
import com.vectordb.db.DemoData;
import com.vectordb.db.DocumentDB;
import com.vectordb.db.VectorDB;
import com.vectordb.util.OllamaClient;
import io.javalin.Javalin;

public class Main {

    private static final int DIMS = 16;
    private static final int PORT = 8080;

    public static void main(String[] args) {
        VectorDB    db     = new VectorDB(DIMS);
        DocumentDB  docDB  = new DocumentDB();
        OllamaClient ollama = new OllamaClient();

        // Load the 20 demo vectors (same as C++ loadDemo)
        DemoData.load(db);

        // Check Ollama at startup (non-fatal)
        boolean ollamaUp = ollama.isAvailable();

        System.out.println("=== VectorDB Engine (Java Edition) ===");
        System.out.println("http://localhost:" + PORT);
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }

        // Start Javalin (replaces cpp-httplib)
        Javalin app = Javalin.create(config -> {
            config.showJavalinBanner = false;
        });

        Routes.register(app, db, docDB, ollama);

        app.start(PORT);

        System.out.println("Server running → open http://localhost:" + PORT + " in your browser");

        // Graceful shutdown on Ctrl+C
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\nShutting down...");
            app.stop();
        }));
    }
}
