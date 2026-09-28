"""
Domain-informed trigger lexicons used by the Tier-1 (rule/statistical)
baseline extractor. These are heuristics, not a trained model — see
COMPONENT2_REPORT.md for the honest limitations of this approach and
scripts/extract_entities_relations_scibert.py for the production path.

Entity type schema follows the SciERC annotation scheme (Luan et al. 2018),
collapsing 'Generic' into 'Other' since generic mentions add no research-gap
signal for Component 5:
    Task, Method, Material, Metric, Other
"""

METRIC_TRIGGERS = {
    "accuracy", "f1", "f-score", "f1-score", "precision", "recall", "auc",
    "auroc", "roc", "bleu", "rouge", "meteor", "perplexity", "score",
    "rate", "error", "loss", "mse", "rmse", "mae", "r-squared", "r2",
    "sensitivity", "specificity", "map", "wer", "cer", "correlation",
    "significance", "p-value", "confidence interval", "odds ratio",
    "likelihood", "log-likelihood", "throughput", "latency", "speedup",
    "coefficient", "variance", "std", "deviation",
}

MATERIAL_TRIGGERS = {
    "dataset", "corpus", "corpora", "benchmark", "data", "images", "samples",
    "patients", "records", "cohort", "survey", "questionnaire", "database",
    "repository", "annotations", "labels", "documents", "tweets", "reviews",
    "articles", "sequences", "genome", "embeddings", "vocabulary", "lexicon",
    "treebank",
}

METHOD_TRIGGERS = {
    "model", "network", "algorithm", "approach", "method", "framework",
    "architecture", "classifier", "transformer", "embedding", "encoder",
    "decoder", "regression", "neural network", "learning", "system",
    "pipeline", "technique", "strategy", "module", "layer", "mechanism",
    "attention", "bert", "lstm", "cnn", "gnn", "rnn", "svm", "gan",
    "autoencoder", "ensemble", "optimizer", "loss function", "estimator",
    "simulation", "protocol", "vaccine", "intervention", "assay", "test",
    "diagnostic", "algorithm",
}

TASK_TRIGGERS = {
    "classification", "detection", "prediction", "translation",
    "segmentation", "generation", "extraction", "recognition", "analysis",
    "tracing", "forecasting", "diagnosis", "summarization", "retrieval",
    "tagging", "parsing", "clustering", "modeling", "identification",
    "screening", "monitoring", "surveillance", "estimation", "inference",
    "understanding", "answering", "reasoning", "annotation",
}

# Verbs/prepositional patterns that license a typed (non-generic) relation.
# Mapped onto the requested Component-2 relation schema.
USED_FOR_VERBS = {"use", "utilize", "employ", "leverage", "apply", "adopt", "propose"}
APPLIED_TO_PREPS = {"to", "on", "for", "in"}
IMPROVES_VERBS = {"improve", "outperform", "boost", "increase", "enhance",
                   "surpass", "beat", "exceed"}
EVALUATED_BY_VERBS = {"evaluate", "measure", "assess", "quantify", "validate",
                       "benchmark", "score"}
ASSOCIATION_VERBS = {"correlate", "associate", "relate", "link", "connect"}

STOPWORD_HEADS = {
    "paper", "study", "work", "result", "results", "finding", "findings",
    "approach", "we", "this", "these", "those", "it", "they", "our",
    "conclusion", "abstract", "introduction",
}

ALL_TYPE_LEXICONS = {
    "Metric": METRIC_TRIGGERS,
    "Material": MATERIAL_TRIGGERS,
    "Method": METHOD_TRIGGERS,
    "Task": TASK_TRIGGERS,
}
