#!/bin/bash
# Sequential scoring runs for all 3 benchmarks with log-space posterior metrics
# Expected total time: ~3.5h (FRANK) + ~12h (TruthfulQA) + ~10h (HaluEval QA) + ...
cd "D:/EvolutionAIry/RANKING_INFERENCE/Experiments"

echo "=== Starting FRANK scoring ==="
echo "Start time: $(date)"
python exp06_frank/run.py --model llama-3.1-8b 2>&1 | tee exp06_frank/scoring_logspace.log
echo "FRANK done at: $(date)"

echo ""
echo "=== Starting TruthfulQA scoring ==="
echo "Start time: $(date)"
python exp05_truthfulqa/run.py --model llama-3.1-8b 2>&1 | tee exp05_truthfulqa/scoring_logspace.log
echo "TruthfulQA done at: $(date)"

echo ""
echo "=== Starting HaluEval scoring ==="
echo "Start time: $(date)"
python exp04_halueval/run.py --model llama-3.1-8b --task all 2>&1 | tee exp04_halueval/scoring_logspace.log
echo "HaluEval done at: $(date)"

echo ""
echo "=== All scoring complete ==="
echo "Finish time: $(date)"
