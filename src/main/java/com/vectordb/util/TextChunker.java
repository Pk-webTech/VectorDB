package com.vectordb.util;

import java.util.*;

public class TextChunker {

    public static List<String> chunk(String text, int chunkWords, int overlapWords) {
        if (text == null || text.isBlank())
            return Collections.emptyList();
        String[] words = text.trim().split("\\s+");
        if (words.length <= chunkWords)
            return List.of(text);

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            chunks.add(String.join(" ", Arrays.copyOfRange(words, i, end)));
            if (end == words.length)
                break;
        }
        return chunks;
    }
}