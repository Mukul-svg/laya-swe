/**
 * LAYA-SWE Extension for Autonomous Coding Agents
 * 
 * Intercepts agent lifecycle events natively:
 * 1. "tool_result": Ingests tool outputs into LAYA-SWE (classifying invariants, symbols, and traces).
 * 2. "context": Prunes bloated middle history while preserving pinned invariants, system prefix, and tool call pairs.
 */

import type { ExtensionAPI, AgentMessage } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  const PROXY_URL = process.env.LAYA_MEMORY_PROXY_URL || process.env.AXIOM_MEMORY_PROXY_URL || "http://127.0.0.1:8080";

  // Auto-Write: Hook tool results (bash, edit, write, read)
  pi.on("tool_result", async (event, ctx) => {
    try {
      const toolName = event.toolName;
      const resultText = typeof event.result === "string" ? event.result : JSON.stringify(event.result);
      if (resultText && resultText.length > 20) {
        // Send asynchronously to background LAYA-SWE engine
        fetch(`${PROXY_URL}/ingest`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            observation: `[TOOL ${toolName}]: ${resultText.slice(0, 1000)}`,
            metadata: { tool: toolName, cwd: ctx.cwd }
          })
        }).catch(() => {});
      }
    } catch (e) {
      // Non-blocking fail-safe
    }
  });

  // Auto-Read & Compaction: Hook context assembly before the LLM prompt
  pi.on("context", async (event, ctx) => {
    const messages = event.messages;
    if (messages.length <= 4) {
      return; // Not enough turns to compact
    }

    try {
      // Find latest intent from user
      let latestIntent = "";
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === "user" && messages[i].content) {
          latestIntent = String(messages[i].content);
          break;
        }
      }

      // Query LAYA-SWE engine for curated evidence & pinned invariants
      const resp = await fetch(`${PROXY_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: latestIntent || "active coding task", top_k: 5 })
      });
      if (!resp.ok) {
        return;
      }

      const data = await resp.json();
      const evidence = data.evidence || "";

      if (evidence) {
        const memoryMessage: AgentMessage = {
          role: "system",
          content: `=== LAYA-SWE ACTIVE MEMORY & EXECUTION CONTEXT ===\n${evidence}\n=================================================`
        };

        // Preserve root user prompt alongside system prompt if present
        const prefix: AgentMessage[] = [messages[0]];
        if (messages.length > 1 && messages[1].role === "user") {
          prefix.push(messages[1]);
        }

        // Safe Tail Slicing: ensure toolResult has its preceding assistant toolCall
        let startIdx = Math.max(prefix.length, messages.length - 4);
        while (startIdx > prefix.length && (messages[startIdx].role === "tool" || (messages[startIdx] as any).role === "toolResult")) {
          startIdx--;
        }
        const tail = messages.slice(startIdx);

        return {
          messages: [...prefix, memoryMessage, ...tail]
        };
      }
    } catch (e) {
      // Fallback: return default messages unmodified if proxy is unavailable
      return;
    }
  });
}
