#!/bin/sh
# Warm the full retrieval path after start so the first visitors don't pay for disk reads.
# The key comes from the service environment and is never printed.
A=/srv/lumbr/artifacts
# 1. Sequentially pre-read the dense index and the BM25 postings/terms/doc values (fast large reads,
#    instead of slow random page faults during queries). Stored fields (.fdt) and term vectors (.tv*) are skipped.
cat "$A/dense_sq_fp16.faiss" > /dev/null 2>&1
for f in "$A"/bm25/*.doc "$A"/bm25/*.pos "$A"/bm25/*.tim "$A"/bm25/*.tip "$A"/bm25/*.dvd "$A"/bm25/*.nvd; do cat "$f" > /dev/null 2>&1; done
# 2. Wait for the app, then run varied queries through the real path.
for i in $(seq 1 60); do curl -sf -m 10 http://127.0.0.1:8000/health >/dev/null && break; sleep 2; done
for q in "how do I reset my password" "what is the refund policy" "who maintains the build pipeline" \
         "how to configure logging" "vacation request process" "onboarding checklist" "incident escalation" \
         "database backup schedule" "expense report deadline" "request access to production" \
         "security training requirements" "where are design docs stored" "release process steps" \
         "code review guidelines" "how to file a bug" "weather in paris" "pasta recipe" \
         "world cup winner" "quantum computing" "mount everest height"; do
  curl -s -o /dev/null -m 120 -X POST http://127.0.0.1:8000/retrieve \
    -H "content-type: application/json" -H "Authorization: Bearer ${LUMBR_BACKEND_KEY}" \
    -d "{\"query\":\"$q\",\"k\":5}"
done
exit 0
