from .mandelbrot import (
    MandelbrotParams,
    mandelbrot_freq,
    mandelbrot_log_freq,
    mandelbrot_pmf,
    fit_mandelbrot_mle,
    fit_mandelbrot_ols_loglog,
    goodness_of_fit,
    compare_distributions,
)
from .rank_utils import (
    RankTable,
    build_rank_table,
    compute_rank_deviations,
    compute_token_level_deviations,
)
from .corpus_utils import (
    DOMAIN_CORPORA,
    REFERENCE_CORPUS,
    get_tokenizer,
    tokenize_text,
    load_corpus_texts,
    tokenize_corpus,
)
from .entity_extraction import (
    EntitySpan,
    EntityGapResult,
    extract_entities,
    align_entities_to_tokens,
    compute_entity_gaps,
    get_grounding_scores,
)
from .aggregation import (
    entity_weighted_mean_delta,
    max_entity_delta,
    proportion_above_threshold,
    log_entity_weighted_mean,
    log_max_entity_delta,
    posterior_entity_weighted_mean,
    aggregate_all,
)
from .benchmark_utils import (
    compute_roc_auc,
    compute_pr_auc,
    bootstrap_ci,
    paired_bootstrap_test,
    compute_f1_at_optimal_threshold,
    compute_cohens_d,
)
from .corpus_scaling import (
    stream_wikipedia_articles,
    build_rank_table_streaming,
    process_full_wikipedia,
)
from .logprob_scoring import (
    ModelConfig,
    ScoringResult,
    score_text_logprobs,
    get_model_config,
    SUPPORTED_MODELS,
    score_batch,
)
from .token_scoring import (
    TokenScores,
    compute_token_scores,
    aggregate_three_modes,
)
