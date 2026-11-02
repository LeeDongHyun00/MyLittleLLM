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
    parser.add_argument("--pipeline", choices=["legacy", "improved", "concise", "preprocessed", "grounded"], default="legacy",
                        help="Improved: hierarchical Qwen retrieval, selected prompt/schema and one source-checked example")
    parser.add_argument("--without-examples", action="store_true", help="Ablate triplet examples in the improved pipeline")
    parser.add_argument("--max-tokens",type=int,help="Output cap: defaults to legacy/concise/preprocessed=384, improved=160")
    parser.add_argument("--admission-track",choices=["논술","학생부교과","학생부종합","실기/실적"])
    parser.add_argument("--lexical-only",action="store_true",help="Preprocessed retrieval without loading neural retrieval models")
    parser.add_argument("--claim-check",action="store_true",help="Experimental neural veto for grounded freeform answers; may reject correct answers")
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
    if args.max_tokens is not None and not 64<=args.max_tokens<=1024:
        parser.error("--max-tokens must be in 64–1024")
    if (args.admission_track or args.lexical_only) and args.pipeline!='preprocessed':
        parser.error("--admission-track and --lexical-only require --pipeline preprocessed")
    if args.pipeline in {'concise','preprocessed'} and (args.mode!='rag' or args.compress_context):
        parser.error("New pipelines support RAG; guarded compression remains an explicit legacy option")
    if args.pipeline=='concise' and args.facts:
        parser.error("Use --pipeline preprocessed or legacy for --facts")
    if args.pipeline=='grounded' and (args.mode!='rag' or args.compress_context or (args.max_tokens is not None and args.max_tokens!=384)):
        parser.error("grounded uses RAG with a fixed 384-token comparison profile and no compression")
    if args.claim_check and args.pipeline!='grounded':parser.error('--claim-check requires --pipeline grounded')
    facts = json.loads(args.facts.read_text()) if args.facts else None
    retriever = bank = None
    if args.pipeline=='grounded':
        from mylittlellm.grounded_assistant import GroundedAssistant
        grounded=GroundedAssistant()
        if args.task=='housing' and args.institution is None:args.institution='한국장학재단'
    output_limit=args.max_tokens or (160 if args.pipeline=='improved' else 384)
    if args.pipeline == "improved":
        from mylittlellm.hierarchical import HierarchicalRetriever
        from mylittlellm.rag_v2 import TripletBank, ask_v2
        from mylittlellm.retrieval_runtime import LowMemoryQwenRetrievalModels
        retriever = HierarchicalRetriever(models=LowMemoryQwenRetrievalModels())
        if not args.without_examples:
            bank = TripletBank(retriever.models)
    if args.pipeline=='preprocessed':
        from mylittlellm.preprocessing import PreprocessedRetriever
        semantic=None
        if not args.lexical_only:
            from mylittlellm.hierarchical import HierarchicalRetriever
            from mylittlellm.retrieval_runtime import LowMemoryQwenRetrievalModels
            semantic=HierarchicalRetriever(models=LowMemoryQwenRetrievalModels())
        retriever=PreprocessedRetriever(semantic=semantic)
    while True:
        try:
            question = args.question or input("질문 (종료: /quit)> ").strip()
            if question == "/quit": break
            if not question: continue
            if args.pipeline=='grounded':
                result=grounded.ask(question,year=args.year,task=args.task,institution=args.institution,
                                    semester=args.semester,facts=facts,base_url=args.base_url,claim_check=args.claim_check)
            elif args.pipeline == "improved":
                result = ask_v2(question, year=args.year, task=args.task, institution=args.institution,
                                semester=args.semester, base_url=args.base_url, retriever=retriever, bank=bank,
                                use_examples=not args.without_examples,max_tokens=output_limit)
            elif args.pipeline in {'concise','preprocessed'}:
                from mylittlellm.preprocessed_assistant import ask_preprocessed
                result=ask_preprocessed(question,year=args.year,task=args.task,institution=args.institution,
                                        semester=args.semester,base_url=args.base_url,retriever=retriever,
                                        pipeline=args.pipeline,max_tokens=output_limit,
                                        admission_track=args.admission_track,facts=facts)
            else:
                result = ask(question, year=args.year, task=args.task, institution=args.institution,
                             semester=args.semester, mode=args.mode, facts=facts, base_url=args.base_url,
                             compress_context=args.compress_context, compression_rate=args.compression_rate,
                             max_tokens=output_limit)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as error:
            raise SystemExit(f"Query failed: {type(error).__name__}: {error}")
        if args.question: break


if __name__ == "__main__": main()
