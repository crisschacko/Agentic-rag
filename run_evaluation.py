from pathlib import Path
from agentic_rag.cli import build_agent
from agentic_rag.evaluation import run_agentic_evaluation

if __name__ == '__main__':
    agent = build_agent('data')
    output = Path('artifacts/agentic_evaluation.csv')
    output.parent.mkdir(exist_ok=True)
    frame = run_agentic_evaluation(agent, output=str(output))
    print(frame[['question', 'decision', 'attempts', 'latency_seconds']].to_string(index=False))
