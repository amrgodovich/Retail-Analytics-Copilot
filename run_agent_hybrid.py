import argparse
import json
from pathlib import Path
from agent.graph_hybrid import app



# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                print("READ LINE:", line)
                yield json.loads(line)

# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def run_batch(input_path, output_path):
    with open(output_path, "a", encoding="utf-8") as f_out:
        for item in read_jsonl(input_path):
            qid = item["id"]
            question = item["question"]
            format_hint = item.get("format_hint", "")

            init_state = {
                "id": qid,
                "question": question,
                "format_hint": format_hint
            }
            # result = app.invoke(init_state)
            # record = {
            #         "id": qid,
            #         "final_answer": result.get("final_answer"),
            #         "sql": result.get("sql_query", "") or "",
            #         "confidence": float(result.get("confidence", 0.0)),
            #         "explanation": result.get("explanation", ""),
            #         "citations": result.get("citations", []),
            #     }
            # print("RECORD:", record)
            try:
                result = app.invoke(init_state)
                record = {
                    "id": qid,
                    "final_answer": result.get("final_answer"),
                    "sql": result.get("sql_query", "") or "",
                    "confidence": float(result.get("confidence", 0.0)),
                    "explanation": result.get("explanation", ""),
                    "citations": result.get("citations", []),
                }
            except Exception as e:
                print(f"Error processing question ID {qid}: {e}")
                record = {
                    "id": qid,
                    "final_answer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": "Failed to process due to rate limit.",
                    "citations": []
                }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush() 



# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    run_batch(args.batch, args.out)
