# PhraseTopicMiner

**Phrase-centric topic mining, clustering, timelines, visualization, and LLM-backed labeling.**

Designed and built by a historian of ideas turned NLP data scientist.

> Instead of treating documents as bags of tokens, PhraseTopicMiner treats **multi-word phrases** as the main carriers of meaning — then builds a **geometric map of ideas** over your corpus.
> 

---

## Why phrase-centric topic modeling?

Most classic topic models (LDA and friends) work at the **word level**:

- They fragment expressions like`topic model`, `topic modeling`, `probabilistic topic models`into partially disconnected tokens.
- They ignore the **multi-word phrases** that humans actually track as conceptual units.
- They give you topics that often read like noisy bags of words.

PhraseTopicMiner starts from a different premise:

> If a text is truly about something, it will keep talking about it and it will do so mostly through recurring noun phrases.
> 

From a linguistic and philosophical point of view:

- Noun phrases encode the **participants and concepts** of a discourse (“latent semantic structure”, “arbitrary royal power”, “freedom under law”).
- Verbs tell you what happens to those concepts, but the **conceptual skeleton** lives in the noun phrases.
- Across a corpus, repeated phrases form **lexical chains** that humans perceive as topics.

PhraseTopicMiner turns that into a pipeline:

- **Phrase mining** gives you the conceptual building blocks.
- **Embedding + clustering** gives you a geometric map of how those concepts
relate.
- **Timelines & visualization** map those clusters back to the sentences
where they live.

Mathematically, the modeling happens in **phrase space**; interpretation and validation happen at the **phrase–sentence interface**.

---

## Core ideas

- **Phrase-centric, not token-centric**
    
    Noun phrases (and some verb phrases) are treated as the primary semantic units.
    
- **Geometric view of topics**
    
    Phrase embeddings + UMAP + clustering → a **topic map** in 2D or higher dimensions.
    
- **Tight link back to text**
    
    Every cluster stays connected to its **supporting sentences** and **positions in documents**.
    
- **Temporal structure**
    
    Topic timelines let you see **when** conceptual constellations appear, grow, overlap, or fade.
    
- **LLM as a hermeneutic assistant (optional)**
    
    The LLM doesn’t replace your judgment; it **proposes labels and explanations** grounded in the cluster evidence.
    

---

## Features at a glance

- 🧩 **Markdown-aware phrase mining**
    - Cleans Markdown (links, footnotes, code fences) before NLP.
    - Extracts **noun phrases** (NP) and **verb phrases** (VP) with rich metadata:
        - document and sentence index
        - phrase kind (`NP` / `VP`)
        - syntactic pattern (`BaseNP`, `NP+PP`, `VerbObj`, `SubjVerb`, …)
        - canonicalized text for counting and modeling.
- 🧭 **Phrase-centric topic modeling**
    - Embeds phrases using:
        - `sentence-transformers` (default),
        - spaCy vectors, or
        - a custom embedding function.
    - Applies optional **PCA denoising**.
    - Uses **UMAP** + **HDBSCAN** by default for robust, shape-aware clustering
    (with KMeans as an alternative).
    - Handles small corpora gracefully (auto-adjusts PCA / UMAP / t-SNE settings).
- 🕰 **Topic timelines**
    - Reconstructs **when** each topic appears in your corpus:
        - simple document index,
        - or approximate **reading time**.
    - Useful for:
        - meeting transcripts,
        - research notebooks over months,
        - intellectual history across decades.
- 📊 **Visualizations**
    - `plot_phrase_bubble_map(core_result, ...)`
        - 2D phrase map with cluster colors and frequency-scaled bubbles.
    - `plot_phrase_treemap(core_result, ...)`
        - Treemap of phrase clusters (“topic constellations”).
    - `plot_topic_timeline(core_result, timeline_result, cluster_id, ...)`
        - Topic intensity over time, linked back to phrases.
- 🤖 **LLM-backed topic labels (experimental)**
    - Optional `TopicLabeler` that takes phrase clusters + representative
    sentences and asks an LLM to propose:
        - a short label,
        - a short description,
        - key phrases to surface in UI.
    - Designed to work with a simple LLM callable, LangChain chat models, or the OpenAI Agents SDK.

---

## Installation

PhraseTopicMiner is on PyPI:

```bash
pip install phrasetopicminer
```

This installs the full stack needed for:

- phrase mining (`spaCy`, `nltk`, `Markdown`, `beautifulsoup4`),
- phrase embeddings (`sentence-transformers`, `scikit-learn`, `umap-learn`, `hdbscan`),
- Plotly visualizations,
- and optional LLM-backed labeling support (via a simple LLM callable, LangChain, or the OpenAI Agents SDK).

> 💡 If you want a lighter installation, you can clone the repo and
> 
> 
> selectively use just the components you need (e.g. only phrase mining).
> 

You’ll also need at least a small English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

(If it isn’t installed, PhraseTopicMiner will try to download it the first time you run the smoke test.)

---

## Quickstart

The minimal example from above, shortened a bit for the README:

```python
import phrasetopicminer as ptm

# 1) A small corpus
docs = [
    "PhraseTopicMiner treats recurring noun phrases as the conceptual skeleton of a text.",
    "Instead of modeling single words, it clusters phrases like 'topic modeling' and "
    " 'customer pain points'.",
    "From these phrase clusters, you get topic maps, timelines, and LLM-backed labels.",
]

# 2) Phrase mining: NP/VP extraction with sentence linkage
miner = ptm.PhraseMiner(spacy_model="en_core_web_sm")

np_counter, vp_counters, phrase_records, sentences_by_doc = miner.mine_phrases_with_types(docs)

print(f"Mined {len(phrase_records)} phrase occurrences")

# 3) Topic modeling in phrase space
modeler = ptm.TopicModeler(
    embedding_backend="sentence_transformers",
    embedding_model="all-MiniLM-L6-v2",
    random_state=42,
)

core = modeler.fit_core(
    phrase_records=phrase_records,
    sentences_by_doc=sentences_by_doc,
    include_kinds={"NP"},
    include_patterns={"BaseNP", "NP+PP", "NP+multiPP"},
    min_freq_unigram=2,
    min_freq_bigram=1,
    min_freq_trigram_plus=1,
    verbose=True,
)

print(core.topics_df[["cluster_id", "size", "top_phrase"]].head())

# 4) Visualize
bubble_fig = ptm.plot_phrase_bubble_map(core)
bubble_fig.show()

```

For a more complete example (including timelines and labeling), see:

- `PhraseTopicMiner.ipynb` in this repository.
- The built-in smoke test:
    
    ```bash
    python -m phrasetopicminer.smoke_test
    ```
    

---

## Core API overview

All public entry points are re-exported at the top level:

```python
import phrasetopicminer as ptm

ptm.PhraseMiner
ptm.TopicModeler
ptm.TopicTimelineBuilder
ptm.TopicLabeler

ptm.plot_topic_timeline
ptm.plot_phrase_bubble_map
ptm.plot_phrase_treemap
ptm.make_datamapplot_static
ptm.make_datamapplot_interactive
```

### Phrase mining

```python
miner = ptm.PhraseMiner(spacy_model="en_core_web_sm", max_docs=None, logger=None)

np_counter, vp_counters, phrase_records, sentences_by_doc = miner.mine_phrases_with_types(
        docs,
        keep_serial_comma=True,
        drop_parenthetical_phrases=True,
    )
```

- `np_counter`: `Counter[str, int]` of canonical noun phrases.
- `vp_counters`: dict of verb-phrase counters per pattern (if enabled).
- `phrase_records`: list of `PhraseRecord`, each with:
    - `phrase`, `canonical`, `kind`, `pattern`,
    - `doc_index`, `sent_index`, character offsets.
- `sentences_by_doc`: list of per-document lists of sentence texts.

### Phrase patterns: how NP / VP extraction works

Under the hood, PhraseTopicMiner uses simple but expressive POS patterns over spaCy’s tagger to define the phrase types it cares about. You’ll see pattern names like `BaseNP`, `NP+PP`, `NP+multiPP`, `VerbObj`, `VerbPP`, `SubjVerb` in the outputs.

We use a tiny tag alphabet for patterns:

- `N` – noun or proper noun (`NOUN`, `PROPN`)
- `A` – adjective (`ADJ`)
- `D` – determiner (`DET`)
- `P` – preposition/adposition (`ADP`)
- `V` – verb (`VERB`)

### BaseNP (Base Noun Phrase): `(A|N)* N`

A base noun phrase is composed of an optional sequence of adjectives or nouns followed by a noun.

Examples:

- `quick fox` (`A N`)
- `brown fox` (`A N`)
- `lazy dog` (`A N`)
- `topic models` (`N N`)

### PP (Prepositional Phrase): `P D* (A|N)* N`

A prepositional phrase starts with a preposition, optionally a determiner, and ends with a base noun phrase.

Examples:

- `over the lazy dog` (`P D A N`)
- `in the archive` (`P D N`)
- `with great power` (`P A N`)
- `under the big blue sky` (`P D A A N`)

### NP (Full Noun Phrase): `BaseNP (PP)*`

A full noun phrase consists of a base noun phrase followed by zero or more prepositional phrases.

Examples:

- `the quick brown fox` (`D A A N`)
- `the fox over the lazy dog` (`D N P D A N`)
- `a big house with red doors` (`D A N P A N`)
- `the tallest building in the city` (`D A N P D N`)

In PhraseTopicMiner you’ll typically use NP patterns like:

- `BaseNP`
- `NP+PP`
- `NP+multiPP`

as filters in `include_patterns`.

### Verb-argument patterns (VP)

Verb phrases are optional but useful when you care about actions and relations, not just entities.

- **VerbObj** – verb + object: roughly `V … N`
    
    Examples:
    
    - `chased the fox` (`V D N`)
    - `delivers services` (`V N`)
    - `proposes solutions` (`V N`)
- **VerbPP** – verb + prepositional phrase: `V PP`
    
    Examples:
    
    - `runs over the hill` (`V P D N`)
    - `jumped into the pool` (`V P D N`)
    - `looked at the stars` (`V P D N`)
- **SubjVerb** – subject–verb: `N V`
    
    Examples:
    
    - `fox jumps` (`N V`)
    - `students struggle` (`N V`)
    - `customers complain` (`N V`)

NPs carry most of the **conceptual load**; VPs are optional “action lenses” that can enrich topic labeling in more process-oriented corpora (e.g. meeting transcripts, procedures, legal obligations).

> Design note – Why spaCy, not NLTK, for POS/NP extraction?
> 
> 
> PhraseTopicMiner uses spaCy for tokenization, tagging, and sentence splitting because it’s fast, robust on modern text, and ships with production-ready English models. If you’re coming from NLTK, the main difference is that you no longer need to manually wire tokenizers + taggers; spaCy gives you a full pipeline and reliable syntactic spans out of the box. NLTK is still great for teaching and low-level experimentation, but spaCy is the default engine behind PhraseMiner.
> 

### Topic modeling

```python
modeler = ptm.TopicModeler(
    embedding_backend="sentence_transformers",   # "sentence_transformers" | "spacy" | "custom"
    embedding_model="all-MiniLM-L6-v2",
    embedding_fn=None,      # used when embedding_backend="custom"
    spacy_nlp=None,         # used when embedding_backend="spacy"
    random_state=42,
)

core = modeler.fit_core(
    phrase_records=phrase_records,
    sentences_by_doc=sentences_by_doc,
    include_kinds={"NP"},
    include_patterns={"BaseNP", "NP+PP", "NP+multiPP"},
    min_freq_unigram=2,
    min_freq_bigram=1,
    min_freq_trigram_plus=1,
    pca_n_components=10,
    cluster_geometry="umap_nd",   # or "umap_2d"
    umap_n_neighbors=10,
    umap_min_dist=0.05,
    umap_cluster_n_components=10,
    clustering_algorithm="hdbscan",  # or "kmeans"
    hdbscan_min_cluster_size=5,
    kmeans_max_clusters=20,
    viz_reducer="tsne_2d",      # "same" | "umap_2d" | "tsne_2d"
    top_n_representatives=10,
    verbose=True,
)
```

`core` is a `TopicCoreResult` with:

- `phrases_df` – one row per phrase (frequency, cluster, coordinates, examples).
- `topics_df` – one row per topic cluster (size, representative phrases, etc.).
- `phrase_sentences` – phrase → example sentences mapping.
- `embedding_geometry` – raw and reduced embeddings.

### Timelines

```python
builder = ptm.TopicTimelineBuilder(
    timeline_mode="reading_time",   # "reading_time" | "document_index" | "token_offset"
    speech_rate_wpm=200,
)

timeline = builder.build(core, sentences_by_doc)
```

`timeline` is a `TopicTimelineResult` used primarily for:

- `plot_topic_timeline(core, timeline, cluster_id=...)`.

### Visualizations

```python
bubble_fig = ptm.plot_phrase_bubble_map(core, max_phrases=200)
treemap_fig = ptm.plot_phrase_treemap(core, max_phrases_per_topic=15)

cluster_id = int(core.topics_df["cluster_id"].iloc[0])
timeline_fig = ptm.plot_topic_timeline(core, timeline, cluster_id=cluster_id)

bubble_fig.show()
treemap_fig.show()
timeline_fig.show()
```

For dense corpora, you can use the **DataMapPlot** helpers (static PNG or HTML)

via `visualization_datamap.py`:

- `make_datamapplot_static(...)`
- `make_datamapplot_interactive(...)`

### Topic labeling with LLMs

Once you have a `TopicCoreResult` from `TopicModeler`, you can attach **human-readable titles and descriptions** to each phrase cluster using `TopicLabeler`.

`TopicLabeler` is deliberately **LLM- and framework-agnostic**. It supports three usage patterns:

1. A simple LLM callable (recommended default)
2. A LangChain `ChatOpenAI` (or similar) wrapped as a callable
3. The OpenAI Agents SDK (`agents`) for agentic workflows + traces

You always give it:

- `core_result`: the `TopicCoreResult` from `TopicModeler.fit_core(...)`
- `sentences_by_doc`: the sentence grid from `PhraseMiner.mine_phrases_with_types(...)`

and get back a `TopicLabelingResult`:

- `labeled_clusters`: full objects with phrases, sentences, and labels
- `labels_by_cluster`: `cluster_id → TopicLabelModel`
- `cluster_name_map`: `cluster_id → title` (ready to plug into plots/treemaps)

### Option A – Minimal, LLM-agnostic callable (no frameworks)

You can keep the dependency surface tiny by passing a plain callable.

The callable can be **sync**:

```python
from phrasetopicminer import TopicLabeler

def simple_llm(prompt: str) -> str:
    # Call your favourite model here (OpenAI, local model, etc.)
    # Return raw text with a JSON object: {"title": "...", "description": "..."}
    raise NotImplementedError
    # e.g., for OpenAI's Python client you might use:
    # resp = client.responses.create(...)
    # return resp.output_text
    #
    # See the OpenAI docs for the concrete call.
    # TopicLabeler will handle JSON parsing / validation.

labeler = TopicLabeler(
    llm=simple_llm,
    max_phrases_per_cluster=25,
    max_sentences_per_cluster=40,
    include_noise=False,
)
```

Or **async**:

```python

async def simple_llm_async(prompt: str) -> str:
    # Same idea, but using an async client call
    return await some_async_model_call(prompt)

labeler = TopicLabeler(
    llm=simple_llm_async,
)

```

TopicLabeler will detect whether llm is sync or async and handle it internally.

To label topics:

- In a **script / non-async context**:

```python
labeling = labeler.label_topics(core_result, sentences_by_doc)
```

- In a **Jupyter notebook** (or any async context):

```python
labeling = await labeler.label_topics_async(core_result, sentences_by_doc)
```

> Tip: in notebooks, call `await labeler.label_topics_async(...)` directly in a cell (do not wrap it inside `%time` or other cell magics, or you’ll get `'await' outside function` errors).
> 

### Option B – LangChain `ChatOpenAI` (or similar)

If you already use LangChain, you can wrap a `ChatOpenAI` (or other chat model)

in a tiny adapter that returns a plain string:

```python
from langchain_openai import ChatOpenAI
from phrasetopicminer import TopicLabeler

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def langchain_llm(prompt: str) -> str:
    resp = chat.invoke(prompt)
    # For ChatOpenAI, resp.content is usually a string
    return resp.content

labeler = TopicLabeler(
    llm=langchain_llm,
    max_phrases_per_cluster=25,
    max_sentences_per_cluster=40,
)

# Script:
labeling = labeler.label_topics(core_result, sentences_by_doc)

# Notebook:
# labeling = await labeler.label_topics_async(core_result, sentences_by_doc)
```

This keeps `TopicLabeler` completely unaware of LangChain; it just sees a

`prompt: str -> str` function.

### Option C – OpenAI Agents SDK (agentic + traces)

If you want **agentic workflows** or to see topic labeling runs in the

OpenAI **Traces** UI, you can use the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/agents/).

Install:

```bash
pip install openai-agents
```

Then:

```python
from agents import Agent  # from the openai-agents package
from phrasetopicminer import TopicLabeler

topic_agent = Agent(
    name="PhraseTopicLabeler",
    model="gpt-4o-mini",
    instructions=(
        "You are a topic labeling assistant. Given key phrases and example "
        "sentences for a single topic, respond ONLY with JSON containing "
        "'title' and 'description'."
    ),
)

labeler = TopicLabeler(
    agent=topic_agent,
    max_phrases_per_cluster=25,
    max_sentences_per_cluster=40,
    include_noise=False,
    labeler_name="agents-sdk-topic-labeler",
)

# In a script, you can still use the sync wrapper:
labeling = labeler.label_topics(core_result, sentences_by_doc)

# In a notebook / async environment:
# labeling = await labeler.label_topics_async(core_result, sentences_by_doc)
```

Under the hood, this path uses:

- `Runner.run(self.agent, prompt)` to call the agent,
- wrapped in a `with trace(...):` block, so each cluster labeling call
    
    appears as a traced workflow.
    

This is the best option if you want PhraseTopicMiner to be part of a

larger **agentic system** (multi-agent workflows, tools, MCPs, etc.) but

don’t want to reinvent the topic labeling step.

---

## What is PhraseTopicMiner good for?

PhraseTopicMiner is not just a way to attach topic labels to documents. It gives you a **thematic summary of a corpus** and shows where different parts of your texts overlap conceptually.

Typical use cases include:

- **Product discovery & UX research**
    - Mine recurring phrases from user interviews, support tickets, and feedback.
    - See clusters like “onboarding friction”, “notification overload”, “manual exports”
    as distinct regions in phrase space.
    - Use timelines to see which themes are emerging vs. stabilizing.
- **Meeting and strategy analysis**
    - Run over meeting transcripts to surface conceptual islands of discussion:
    roadmap decisions, technical debt, specific customer pain points.
    - Track how topics evolve across sprints or quarters.
- **Research & literature mapping**
    - Apply to abstracts, introductions, or sections of papers in a subfield.
    - Discover constellations of methods, problem settings, and evaluation strategies.
    - Use the phrase map as a **conceptual overview** of a research area.
- **Education & curriculum design**
    - Analyze lecture notes, assignments, and forum posts.
    - See which concepts cluster together, where students struggle, and how the
    “conceptual difficulty landscape” changes over a course.
- **Intellectual history & history of ideas**
    - Mine multi-word vocabularies like “freedom under law”, “arbitrary royal power”,
    “rights of the people”, “religious authority” across archives.
    - Use timelines to track how different **constellations of phrases** rise, overlap,
    or fade over years and decades.

Because topics are defined as **clusters of phrases**, each of which is tied back to sentences and documents, PhraseTopicMiner makes it easy to answer questions like:

- “Which sentences in which documents contribute to this conceptual region?”
- “Where do two topic constellations overlap in the corpus?”
- “How does this theme appear and transform over time?”

### 💭 Theoretical background (for the curious) – NPs as carriers of “aboutness”

> Most topic models work at the level of single words. PhraseTopicMiner starts from a different bet:
> 
> 
> if a text is *about* something, it will keep saying it – and it will say it mostly with **noun phrases**.
> 
> In discourse theory and functional linguistics, “aboutness” is usually carried by **participants** in a clause – the entities, ideas, and institutions we keep talking about. These are overwhelmingly realized as noun phrases: *“probabilistic topic models”*, *“constitutional limits”*, *“customer pain points”*, *“problem solving”*. Verbs tell us what happens to these entities; noun phrases tell us **what the conversation is actually about**.
> 
> PhraseTopicMiner treats these recurring noun phrases as points in a semantic space and clusters them into **conceptual constellations**. Each cluster is a candidate “topic”: not in the sense of a hidden variable in a generative model, but as a stable region in the text’s **concept-geometry** – the way ideas group and recur.
> 
> Crucially, the system never forgets the sentences. Every phrase is anchored back to its original sentences, so each cluster can be unfolded into the **discursive context** that gave rise to it. The math happens in phrase space; the *interpretation* happens at the phrase–sentence interface. In Collingwood’s terms, these NP clusters are the recurring answers that reveal the underlying question-space of a corpus: the problems a community keeps circling around, in its own language.
> 

---

## Roadmap

The 0.1.x series focuses on:

- Stabilizing the core API (`PhraseMiner`, `TopicModeler`, timelines, viz).
- Tightening small-corpus behavior and defaults.
- Improving docs and example notebooks.

Planned future work:

- First-class support for non-English languages (custom spaCy models).
- Better LLM-based labeling via the OpenAI Agents SDK and custom prompts.
- Integration with RAG / knowledge-graph pipelines (export topic graphs, etc.).
- A small gallery of “recipes” for:
    - product discovery (user interviews, support tickets),
    - research idea mapping (papers, abstracts),
    - intellectual history (archives across decades).

---

## Contributing

Issues and pull requests are welcome.

- **Bug reports**: please include a minimal reproducible example (even 2–3
    
    short docs are enough).
    
- **Feature requests**: describe your use-case (research, product analytics,
    
    history of ideas, etc.) so we can keep the library grounded in real workflows.
    

---

## License

MIT License — see `LICENSE` for details.

If you use PhraseTopicMiner in academic work, you’re encouraged (but not required) to cite the project and, where relevant, the associated work on **conceptual history and phrase-centric topic modeling**.

---

## About the author

**Ahmad Hashemi** is an NLP data scientist and Principal NLP engineer with a DPhil (PhD) in Philosophy from the University of Oxford, specializing in intellectual history and the history of political ideas.

PhraseTopicMiner grew out of a long-standing question:

> How can we give machines a more human way of “seeing” the conceptual structure of a corpus, not just as statistics over words, but as evolving constellations of phrases?
> 

If this resonates with your work (research or product), feel free to reach out via GitHub or LinkedIn.