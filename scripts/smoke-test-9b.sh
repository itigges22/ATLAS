#!/bin/bash
# Smoke test for Qwen3.5-9B deployment
# Tests: generation, embeddings, prompt formatting, Lens scoring
set -e
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

POD_IP=$(kubectl get pods -n atlas -l app=llama-server -o jsonpath='{.items[0].status.podIP}')
LLAMA_URL="http://${POD_IP}:8000"
RAG_IP=$(kubectl get pods -n atlas -l app=geometric-lens -o jsonpath='{.items[0].status.podIP}')
RAG_URL="http://${RAG_IP}:8001"

echo "=== Qwen3.5-9B Smoke Test ==="
echo "  llama-server: $LLAMA_URL"
echo "  geometric-lens: $RAG_URL"
echo ""

# Test 1: Health check
echo "--- Test 1: Health Check ---"
curl -s "$LLAMA_URL/health" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Status: {d.get(\"status\",\"unknown\")}')"
echo ""

# Test 2: Basic generation (nothink)
echo "--- Test 2: Generation (nothink) ---"
START=$(date +%s%N)
RESPONSE=$(curl -s "$LLAMA_URL/completion" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a Python function that returns the sum of two numbers. Just the function, no explanation.<|im_end|>\n<|im_start|>assistant\n",
    "n_predict": 256,
    "temperature": 0.0,
    "seed": 42,
    "stop": ["<|im_end|>"]
  }')
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))
TOKENS=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tokens_predicted', 0))")
CONTENT=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('content', '')[:200])")
echo "Tokens: $TOKENS | Time: ${ELAPSED}ms | tok/s: $(python3 -c "print(f'{$TOKENS / ($ELAPSED/1000):.1f}' if $ELAPSED > 0 else 'N/A')")"
echo "Content: $CONTENT"
echo ""

# Test 3: Generation with thinking
echo "--- Test 3: Generation (thinking) ---"
START=$(date +%s%N)
RESPONSE=$(curl -s "$LLAMA_URL/completion" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a Python function that checks if a string is a palindrome. Include edge cases.<|im_end|>\n<|im_start|>assistant\n<|im_start|>think\n",
    "n_predict": 1024,
    "temperature": 0.6,
    "seed": 42,
    "stop": ["<|im_end|>"]
  }')
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))
TOKENS=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tokens_predicted', 0))")
echo "Tokens: $TOKENS | Time: ${ELAPSED}ms | tok/s: $(python3 -c "print(f'{$TOKENS / ($ELAPSED/1000):.1f}' if $ELAPSED > 0 else 'N/A')")"
HAS_THINK=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); c=d.get('content',''); print('YES' if '</think>' in c else 'NO')")
echo "Has </think> tag: $HAS_THINK"
echo ""

# Test 4: Embeddings
echo "--- Test 4: Embeddings ---"
EMB_RESPONSE=$(curl -s "$LLAMA_URL/embedding" \
  -H "Content-Type: application/json" \
  -d '{"content": "def add(a, b): return a + b"}')
DIM=$(echo "$EMB_RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
emb = d.get('embedding', [])
if emb and isinstance(emb[0], list):
    emb = emb[0]
print(f'Dimension: {len(emb)}')
norm = sum(x*x for x in emb) ** 0.5
print(f'L2 norm: {norm:.4f}')
print(f'First 3: {emb[:3]}')
")
echo "$DIM"
echo ""

# Test 5: Slots check
echo "--- Test 5: Slots ---"
curl -s "$LLAMA_URL/slots" | python3 -c "
import json, sys
slots = json.load(sys.stdin)
for s in slots:
    print(f'  Slot {s[\"id\"]}: state={s.get(\"state\",\"?\")} n_ctx={s.get(\"n_ctx\",0)}')
"
echo ""

# Test 6: RAG API health (if available)
echo "--- Test 6: RAG API ---"
RAG_HEALTH=$(curl -s --max-time 5 "$RAG_URL/health" 2>/dev/null || echo '{"status":"unreachable"}')
echo "$RAG_HEALTH" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'RAG API: {d.get(\"status\",\"unknown\")}')" 2>/dev/null || echo "RAG API: unreachable"
echo ""

echo "=== Smoke Test Complete ==="
