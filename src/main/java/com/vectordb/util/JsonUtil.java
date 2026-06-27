package com.vectordb.util;

import java.util.ArrayList;
import java.util.List;

public class JsonUtil {

    /** Escape and quote a string for JSON output */
    public static String jS(String s) {
        if (s == null)
            return "\"\"";
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> sb.append(c);
            }
        }
        return sb.append('"').toString();
    }

    /** Serialize a float list to a JSON array string */
    public static String jVec(List<Float> v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.size(); i++) {
            if (i > 0)
                sb.append(',');
            sb.append(String.format("%.4f", v.get(i)));
        }
        return sb.append(']').toString();
    }

    /** Parse a comma-separated float string (from query param) */
    public static List<Float> parseVec(String s) {
        List<Float> result = new ArrayList<>();
        if (s == null || s.isBlank())
            return result;
        for (String tok : s.split(",")) {
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

    /** Extract a JSON string field value from raw JSON body */
    public static String extractStr(String body, String key) {
        if (body == null)
            return "";
        int p = body.indexOf('"' + key + '"');
        if (p < 0)
            return "";
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t'))
            p++;
        if (p >= body.length() || body.charAt(p) != '"')
            return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"')
                break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                switch (body.charAt(p)) {
                    case '"' -> result.append('"');
                    case '\\' -> result.append('\\');
                    case 'n' -> result.append('\n');
                    case 'r' -> result.append('\r');
                    case 't' -> result.append('\t');
                    default -> result.append(body.charAt(p));
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    /** Extract a JSON integer field value from raw JSON body */
    public static int extractInt(String body, String key, int def) {
        if (body == null)
            return def;
        int p = body.indexOf('"' + key + '"');
        if (p < 0)
            return def;
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t'))
            p++;
        try {
            return Integer.parseInt(body.substring(p).replaceAll("[^\\d-].*", "").trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    /**
     * Extract float[] embedding from JSON body
     * {"metadata":"..","category":"..","embedding":[..]}
     */
    public static List<Float> extractEmbedding(String body) {
        if (body == null)
            return new ArrayList<>();
        int p = body.indexOf("\"embedding\"");
        if (p < 0)
            return new ArrayList<>();
        int start = body.indexOf('[', p);
        if (start < 0)
            return new ArrayList<>();
        int end = body.indexOf(']', start);
        if (end < 0)
            return new ArrayList<>();
        return parseVec(body.substring(start + 1, end));
    }
}
