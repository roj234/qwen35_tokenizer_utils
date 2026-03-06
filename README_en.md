
# Tokenizer Analyzer & Patcher

This project provides a set of diagnostic and patching tools to identify and fix "under-trained tokens" in BPE tokenizers (specifically for Qwen3.5).

> If you are not chinese user, this project's RESULT (e.g. `tokenizer_patched.json`) might NOT useful for you.  
> However, its algorithm and thought maybe useful.  
> For how to use (rather than implement details) please refer to the Chinese version of this documentation.

---

## What Problem Does This Solve?

### 1. The "Bad Token" Phenomenon
Large Language Models (LLMs) often have vocabulary entries (tokens) that were rarely encountered during pre-training. For Qwen3.5, certain long Chinese tokens act as "black boxes." When the model encounters these, it fails to understand their internal structure, leading to **Associative Hallucinations** (fabricating content that doesn't exist in the prompt).

**Example Tokens that trigger hallucinations in Qwen3.5:**
- `开通天眼生意通银牌及以上会员`
- `认证成功后可编辑`
- `转贴或以其他方式`
- `加分后可超过约`
- It is not only affect Chinese tokens, there are also some Thai tokens (or other) affected

#### Adversarial Samples
You can view and locally test [adversarial samples (Chinese)](spring_ad.md).  
> Experiments show that even on 400B models, these high-PPL "toxic" tokens still cause severe "associative hallucinations" (fabricating keywords that do not exist in the source text).

### 2. Systematic Improvement
By identifying these tokens and **removing their "merges" rules**, we force the tokenizer to decompose these "bad tokens" into smaller, well-trained sub-tokens.
*   **Result:** The model regains the ability to reason about the text through its components.
*   **Safety:** Unlike deleting the vocabulary index, removing merge rules ensures the model can still generate these tokens if it *knows*, but will prioritize using sub-units on tokenize process.

---

## Core Algorithms (Methodology)

The project evolves through three iterations of detection logic to identify problematic tokens:

### Algorithm 1.0: Semantic Alignment Detection
*   **Logic:** It calculates the **Cosine Similarity** between the Embedding of a "long token" and the average Embedding of its "decomposed sub-tokens" (segmented by a (not really™) more robust reference model).
*   **Assumption:** If the high-level token's vector deviates significantly from the meaning of its constituent parts, it is likely under-trained.

### Algorithm 2.0: Logical Decomposition Test
*   **Logic:** It prompts the model to explain the internal characters of a token using a structured JSON format.
*   **Assumption:** If a model cannot logically break down a token into its individual characters, the token is a "black box" to the model.

### Algorithm 2.1: Last Token PPL (Perplexity) Analysis
*   **Logic:** Building on 2.0, this algorithm calculates the **Perplexity (PPL)** of the model when predicting the closing bracket `]` of the expected JSON response.
*   **Refinement:** A high PPL indicates that the model is "confused" or "uncertain" about the token's internal structure. This provides a quantifiable metric to rank which tokens are the most "toxic" to the model's performance.

---

## The Patching Strategy

Instead of modifying the `vocab` (which would break model compatibility), this tool modifies `merges` section in `tokenizer.json`:
1.  **Force Decomposition:** By deleting the merge rule that creates the "bad token," the tokenizer naturally defaults to the next best (and usually better-trained) sub-tokens.
2.  **Zero-Shot Recovery:** Experiments show that even without fine-tuning, simply forcing the model to see the "constituent parts" of a previously problematic phrase can instantly eliminate hallucinations.
