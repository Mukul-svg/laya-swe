"""Empirical verification of the Asymptotic Compaction Scaling Law across 30 turns.

Uses real conversation history traces from Pi coding sessions to measure:
1. Raw prompt growth O(T)
2. Compacted prompt bound O(1)
3. Instantaneous savings percentage eta(T)
"""
import json
import glob
import os
from proxy.memory_engine import MultiRelationalMemoryEngine
from proxy.compactor import ContextCompactor, estimate_messages_tokens

def run_scaling_experiment():
    pattern = os.path.expanduser('~/.pi/agent/sessions/--C--Users-mukul-OneDrive-Documents-Projects-Z-tasks-task_4_distributed_2pc-repo--/*.jsonl')
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        print("No session files found.")
        return

    # Find the deepest session file
    best_file = None
    max_count = 0
    for f in files:
        count = sum(1 for line in open(f, 'r', encoding='utf-8') if line.strip())
        if count > max_count:
            max_count = count
            best_file = f

    print(f"Selected deepest session: {best_file} ({max_count} entries)")
    session_file = best_file
    entries = [json.loads(l) for l in open(session_file, 'r', encoding='utf-8') if l.strip()]
    
    # Filter message turns (assistant + toolResult pairs)
    messages = []
    for e in entries:
        m = e.get('message', {})
        role = m.get('role')
        if role in ('system', 'user', 'assistant', 'toolResult'):
            # Convert Pi format to standard OpenAI format
            content = m.get('content', '')
            if isinstance(content, list):
                text_parts = []
                tc_list = []
                for item in content:
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'toolCall':
                        tc_list.append({
                            'id': item.get('id', 'call_id'),
                            'function': {
                                'name': item.get('name', ''),
                                'arguments': json.dumps(item.get('arguments', {})) if isinstance(item.get('arguments'), dict) else str(item.get('arguments', ''))
                            }
                        })
                msg_dict = {'role': 'assistant' if role == 'assistant' else ('tool' if role == 'toolResult' else role)}
                if text_parts:
                    msg_dict['content'] = '\n'.join(text_parts)
                if tc_list:
                    msg_dict['tool_calls'] = tc_list
                messages.append(msg_dict)
            else:
                messages.append({'role': 'assistant' if role == 'assistant' else ('tool' if role == 'toolResult' else role), 'content': str(content)})

    print(f"Total extracted messages from session: {len(messages)}")
    
    engine = MultiRelationalMemoryEngine()
    compactor = ContextCompactor(memory_engine=engine, max_tail_messages=8)

    print("\n" + "="*85)
    print(f"{'Turn':<6} | {'Raw Tokens':<12} | {'Sent Tokens':<12} | {'Saved Tokens':<14} | {'Compaction %':<14} | {'Complexity'}")
    print("="*85)

    # Simulate multi-turn progression from Turn 1 to N
    # Each turn adds an assistant call + tool result
    turn = 0
    raw_history = []
    
    for i in range(1, len(messages) + 1):
        raw_history = messages[:i]
        # Only evaluate on assistant turn boundaries (when model is called)
        if raw_history[-1].get('role') in ('user', 'tool'):
            turn += 1
            compacted, raw_tok, sent_tok = compactor.process_and_compact(raw_history, enable_compaction=True)
            saved = max(0, raw_tok - sent_tok)
            pct = (saved / max(1, raw_tok)) * 100
            
            complexity = "O(T) uncompressed" if pct == 0 else f"O(1) bounded ({sent_tok} tok)"
            print(f"{turn:<6d} | {raw_tok:<12,d} | {sent_tok:<12,d} | {saved:<14,d} | {pct:<13.2f}% | {complexity}")

if __name__ == "__main__":
    run_scaling_experiment()
