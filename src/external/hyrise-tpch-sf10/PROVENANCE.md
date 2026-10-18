# hyrise TPC-H SF=10 real index sizes

**Source:** hyrise/index_selection_evaluation (the open-source platform behind
Kossmann et al., "Magic mirror in my hand, which is the best in the land?",
PVLDB 13(11), 2020). https://github.com/hyrise/index_selection_evaluation

**File:** benchmark_results/tpch_wo_2_17_20/cophy_ampl_data_tpch_3_1.txt
(CoPhy AMPL data; TPC-H scale factor 10; queries wo 2,17,20; max index width 3)

**What:** 1,215 candidate indexes with real HypoPG size estimates (PostgreSQL
hypothetical-index sizes on actual SF=10 data) and their column lists.

**Extracted to:** index_sizes.json  [{id, size_bytes, columns, n_cols}, ...]

**Key facts:** median 1-col 209 MB, 2-col 2,351 MB, 3-col 4,076 MB; skew
(excl. nation/region) ~2,437x; single-column excl. micro ~632x.
