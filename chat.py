"""Run one scoped research query, or an interactive session in the same scope."""
import argparse
import json
from pathlib import Path
from mylittlellm.assistant import ask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--task", choices=["admissions", "housing", "scholarship"], default="admissions")
    parser.add_argument("--institution", choices=["서울대학교", "부산대학교", "경희대학교", "한국장학재단", "교육부"])
    parser.add_argument("--semester", type=int, choices=[1, 2])
    parser.add_argument("--mode", choices=["rag", "closed"], default="rag")
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--pipeline", choices=["legacy", "improved"], default="legacy",
                        help="Improved: hierarchical Qwen retrieval, selected prompt/schema and one source-checked example")
    parser.add_argument("--without-examples", action="store_true", help="Ablate triplet examples in the improved pipeline")
    parser.add_argument("--compress-context", action="store_true", help="Use guarded LLMLingua-2; requires training dependencies and assets")
    parser.add_argument("--compression-rate", type=float, default=.5)
    args = parser.parse_args()
    if args.task == "admissions" and not args.institution:
        parser.error("--institution is required for admissions")
    if args.task == "housing" and args.semester is None:
        parser.error("--semester is required for housing")
    if args.pipeline == "improved" and (args.mode != "rag" or args.facts or args.compress_context):
        parser.error("--pipeline improved supports RAG without --facts or --compress-context; the legacy pipeline supports these existing features")
    if args.without_examples and args.pipeline != "improved":
        parser.error("--without-examples requires --pipeline improved")
    facts = json.loads(args.facts.read_text()) if args.facts else None
    retriever = bank = None
    if args.pipeline == "improved":
        from mylittlellm.hierarchical import HierarchicalRetriever
        from mylittlellm.rag_v2 import TripletBank, ask_v2
        from mylittlellm.retrieval_runtime import LowMemoryQwenRetrievalModels
        retriever = HierarchicalRetriever(models=LowMemoryQwenRetrievalModels())
        if not args.without_examples:
            bank = TripletBank(retriever.models)
    while True:
        try:
            question = args.question or input("질문 (종료: /quit)> ").strip()
            if question == "/quit": break
            if not question: continue
            if args.pipeline == "improved":
                result = ask_v2(question, year=args.year, task=args.task, institution=args.institution,
                                semester=args.semester, base_url=args.base_url, retriever=retriever, bank=bank,
                                use_examples=not args.without_examples)
            else:
                result = ask(question, year=args.year, task=args.task, institution=args.institution,
                             semester=args.semester, mode=args.mode, facts=facts, base_url=args.base_url,
                             compress_context=args.compress_context, compression_rate=args.compression_rate)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as error:
            raise SystemExit(f"Query failed: {type(error).__name__}: {error}")
        if args.question: break


if __name__ == "__main__": main()
