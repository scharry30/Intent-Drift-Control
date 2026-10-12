"""Public benchmark source URLs and prompt-field metadata."""

SOURCES = {
    "strongreject": (
        "csv",
        "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
        "forbidden_prompt",
        "harmful",
    ),
    "xstest": (
        "csv",
        "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
        "prompt",
        "by_label",
    ),
    "or_bench_hard": (
        "csv",
        "https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-hard-1k.csv",
        "prompt",
        "benign",
    ),
    "jbb_behaviors": (
        "csv",
        "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv",
        "Goal",
        "harmful",
    ),
    "math500": (
        "jsonl",
        "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
        "problem",
        "reasoning",
    ),
    "sorry_bench": (
        "jsonl",
        "https://huggingface.co/datasets/sorry-bench/sorry-bench-202406/resolve/main/question.jsonl",
        "turns",
        "harmful",
    ),
    "wildjailbreak": (
        "tsv",
        "https://huggingface.co/datasets/allenai/wildjailbreak/resolve/main/eval/eval.tsv",
        "adversarial",
        "harmful",
    ),
    "gpqa_diamond": (
        "csv",
        "https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_diamond.csv",
        "Question",
        "reasoning",
    ),
}
