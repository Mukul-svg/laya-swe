/**
 * LAYA-SWE Extension for Autonomous Coding Agents (Pi Coding Agent Native Lifecycle Integration)
 * 
 * Intercepts agent lifecycle events natively:
 * 1. "tool_result": Ingests tool outputs with explicit 'tool' provenance into LAYA-SWE.
 * 2. "context": Ingests unrecorded user turns ('user' provenance), extracts active intent,
 *    and applies batch-and-freeze in-place masking for older tool observations while keeping
 *    the conversation prefix stable.
 */

import type { ExtensionAPI, AgentMessage } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const PROXY_URL = process.env.LAYA_MEMORY_PROXY_URL || process.env.AXIOM_MEMORY_PROXY_URL || "http://127.0.0.1:8080";
  const ingestedMessageHashes = new Set<string>();

  // Helper to hash/deduplicate observations
  function getShortHash(str: string): string {
    return str.slice(0, 80).trim();
  }

  // Auto-Write: Hook tool results (bash, edit, write, read) with tool provenance
  pi.on("tool_result", async (event, ctx) => {
    try {
      const toolName = event.toolName || "tool";
      // Pi exposes content, details, and isError on tool_result
      const rawContent = (event as any).content ?? (event as any).result ?? "";
      const resultText = typeof rawContent === "string" ? rawContent : JSON.stringify(rawContent);

      if (resultText && resultText.length > 20) {
        const hashKey = `tool:${toolName}:${getShortHash(resultText)}`;
        if (!ingestedMessageHashes.has(hashKey)) {
          ingestedMessageHashes.add(hashKey);
          // Explicitly tag role as 'tool' to enforce provenance-gated pinning security
          await fetch(`${PROXY_URL}/ingest`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              observation: `[OBSERVED TOOL OUTPUT - ${toolName}]: ${resultText.slice(0, 1200)}`,
              metadata: { tool: toolName, cwd: ctx?.cwd, isError: Boolean((event as any).isError) },
              role: "tool"
            })
          }).catch(() => {});
        }
      }
    } catch {
      // Non-blocking fail-safe
    }
  });

  // Auto-Read & Compaction: Hook context assembly before the LLM prompt
  pi.on("context", async (event, ctx) => {
    const messages = event.messages;
    if (!messages || messages.length <= 2) {
      return;
    }

    try {
      // 1. Ingest unrecorded user messages with 'user' provenance so invariants can be pinned!
      for (const m of messages) {
        if (m.role === "user" && m.content) {
          const userText = String(m.content).trim();
          const hashKey = `user:${getShortHash(userText)}`;
          if (!ingestedMessageHashes.has(hashKey) && userText.length > 10) {
            ingestedMessageHashes.add(hashKey);
            await fetch(`${PROXY_URL}/ingest`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                observation: `[USER GOAL]: ${userText}`,
                metadata: { role: "user" },
                role: "user"
              })
            }).catch(() => {});
          }
        }
      }

      // 2. Extract active query from latest user or tool intent
      let activeQuery = "";
      for (let i = messages.length - 1; i >= 0; i--) {
        const m = messages[i];
        if (m.role === "user" && m.content) {
          activeQuery = String(m.content);
          break;
        } else if (((m as any).role === "toolResult" || m.role === "tool") && m.content) {
          activeQuery = String(m.content).slice(0, 300);
          break;
        }
      }

      // 3. Query LAYA-SWE engine for curated evidence & pinned invariants
      const resp = await fetch(`${PROXY_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: activeQuery || "active coding task", top_k: 4 })
      });
      if (!resp.ok) {
        return;
      }

      const data = await resp.json();
      const evidence = data.evidence || "";
      if (!evidence) {
        return;
      }

      // 4. In-Place History Compaction with Stable Placeholders
      // Rather than dropping middle turns (which thrashes KV cache), compact older tool results in-place
      const tailWindowSize = 4;
      const tailStart = Math.max(0, messages.length - tailWindowSize);

      const compactedMessages = messages.map((m, idx) => {
        const isToolMsg = (m as any).role === "toolResult" || m.role === "tool";
        if (idx < tailStart && isToolMsg && m.content) {
          const text = String(m.content);
          if (text.length > 600) {
            const omitted = text.length - 400;
            return {
              ...m,
              content: text.slice(0, 200) + `\n[Tool output compacted: ${omitted} chars omitted. Re-run tool to view full output]\n` + text.slice(-200)
            };
          }
        }
        return m;
      });

      // 5. Append Curated System-1 Memory at Prompt Tail (Cache-Friendly & Privilege-Safe)
      const executionContextMessage: AgentMessage = {
        role: "user", // Kept in unprivileged user/context role
        content: `=== LAYA-SWE SYSTEM-1 MEMORY & EXECUTION CONTEXT ===\n${evidence}\n=====================================================`
      };

      return {
        messages: [...compactedMessages, executionContextMessage]
      };
    } catch {
      // Fallback: return default messages unmodified if proxy is unavailable
      return;
    }
  });
}
