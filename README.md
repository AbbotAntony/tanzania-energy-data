# Tanzania Natural Gas & Petroleum Data Scraper

A reproducible pipeline that collects Tanzanian data on natural gas utilisation, heavy-fuel-oil (HFO) imports, fuel prices and power generation, plus energy news articles. The data supports energy-economics research and machine-learning work, including NLP analysis of how Tanzania's media covers natural gas and energy. Tables from public documents are turned into tidy CSV and Stata `.dta` files, and every value keeps full provenance back to its source document.

## Setup (Windows, PowerShell)

```powershell
cd "C:\Users\HomePC\Desktop\web scraping"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# optional: override the contact address in the User-Agent
# (default: 92087158+AbbotAntony@users.noreply.github.com)
$env:TZ_SCRAPER_CONTACT = "92087158+AbbotAntony@users.noreply.github.com"
```

## Running

```powershell
.\.venv\Scripts\python.exe -m pipeline list-sources
.\.venv\Scripts\python.exe -m pipeline run --source ewura              # list -> download -> parse -> export
.\.venv\Scripts\python.exe -m pipeline run --source ewura --only list  # just show what is published
.\.venv\Scripts\python.exe -m pipeline run --source ewura --refresh    # ignore the HTTP cache
.\.venv\Scripts\python.exe -m pipeline qa                              # missing-year and discrepancy reports
.\.venv\Scripts\python.exe -m pytest                                    # offline tests
```

The first EWURA run downloads about 110 PDFs (about 300 MB, roughly 10 minutes at
3–5 s per request). It then spends several more minutes extracting text from them.
Later runs use the HTTP cache and the PDF-text cache and finish in seconds.

## Layout

| Path | Contents |
|---|---|
| `pipeline/` | HTTP client (robots.txt, 3–5 s delay, retries, cache), provenance records, raw storage, CSV/.dta export, QA, CLI |
| `scrapers/` | one module per source (`ewura.py` so far) |
| `parsers/` | listing-page parser, PDF text extraction and glyph decoding, text-table engine, one parser per report type |
| `tests/` | pytest; fixtures are saved HTML pages and page text cut from the real PDFs |
| `data/raw/<source>/<YYYY-MM-DD>/` | unchanged downloads plus `manifest.jsonl` (url, title, sha256, retrieval time) |
| `data/processed/` | output datasets and QA reports |
| `logs/` | one log per source per day; failed URLs are logged at WARNING |
| `nlp/` | NLP analysis of the news corpus, with its own environment (see [NLP analysis](#nlp-analysis-how-daily-news-covers-natural-gas)) |

## Output format

Each dataset is tidy: **one row per published number**. Main columns:

* `year`, `month`, `fy`: `fy` is the Tanzanian fiscal year (July–June), e.g.
  `2023/24`. **For FY data `year` is the year the FY ends** (2023/24 → 2024).
  For calendar-year data `fy` is empty.
* `variable`, `category`: what is measured, e.g. `gas_consumption` / `Power`.
* `value_<unit>`: the number in the unit **as published**, with one column per
  unit (`value_mmscf`, `value_mmscfd`, `value_pct`, `value_litres`,
  `value_tzs_per_litre`, `value_count`). Nothing is converted. `conversion_note`
  is reserved for any future documented conversion.
* `value_as_printed`: the exact string in the source.
* Provenance: `source`, `url`, `doc_title`, `pub_date` (listing date),
  `retrieved_date`, `pdf_page` (1-based PDF page), `table_no`.
* `qa_flag`: problems detected; values are never corrected (see below).
* `x_*`: dataset-specific keys, e.g. `x_seller` (TPDC/PAET/M&P), `x_market`
  (Local/Transit), `x_port`, `x_price_basis`, `x_report_fy` (which edition), `x_table_caption`.

The same figure often appears in several report editions. **All editions are
kept** (filter on `x_report_fy` or `doc_title`). Differences between them are
listed in `qa_discrepancies.csv` rather than resolved.

### `qa_flag` values

| Flag | Meaning |
|---|---|
| `total_ne_sum_of_rows(sum=…)` | the printed TOTAL differs from the sum of its rows by more than 0.5 % |
| `total_ne_sum_of_sellers(sum=…)` | TPDC + PAET + M&P differs from the printed Total column |
| `short_row_right_aligned` | the row has fewer numbers than columns; values were assigned to the right-most columns |
| `blank_in_source` | the cell is printed blank; the row is kept with an empty value |
| `text_glyph_decoded` | the page's text was recovered from glyph codes (see below); worth spot-checking |
| `fewer_cells_than_columns_left_aligned` | cap price row is missing trailing columns (e.g. no kerosene at Tanga) |

## Sources

Verified on 2026-10-15. Robots.txt was checked for every site.

| Source | Status | Listing URL |
|---|---|---|
| **EWURA** | **implemented** | `https://www.ewura.go.tz/publications/performance-report`, `/publications/petroleum-price` (use `www.`; the bare domain times out) |
| **Bank of Tanzania** | **implemented** | Zonal Economic Report `https://www.bot.go.tz/Publications/Filter/36`; Monthly Economic Review `/Publications/Filter/1` |
| **TPDC** | **implemented (OCR)** | `https://www.tpdc.co.tz/documents/reports` (audited financial statements FY2015–FY2025) |
| **Ministry of Energy** | **implemented** | `https://www.nishati.go.tz/documents/useful-documents` |
| PBPA | surveyed, skipped | `https://www.pbpa.go.tz/publications/tender-results-for-supply-of-petroleum-products` |
| **TANESCO** | **implemented (Playwright)** | `https://www.tanesco.co.tz/annual-report` |
| **Daily News** | **implemented** | WordPress JSON API `https://dailynews.co.tz/wp-json/wp/v2/posts?search=…` |
| **The Citizen** | **implemented (headlines only)** | `https://www.thecitizen.co.tz/sitemap.xml` (paywalled; search disallowed by robots.txt) |
| **NBS** | **implemented (Energy Balance 2022, Excel + PDFs)** | `https://www.nbs.go.tz/statistics/topic/energy-balance` |
| PURA, GASCO | not built | no data publications, news pages only |

### EWURA datasets

| File | Contents | Coverage |
|---|---|---|
| `ewura_gas` | gas supply by processing plant, production by field, consumption by customer category (total and by seller), pipeline and processing-plant capacity utilisation, customer counts | supply FY2018/19–FY2024/25; consumption by category FY2020/21–FY2024/25; pipeline utilisation FY2019/20–FY2024/25 |
| `ewura_gas_narrative` | sentences mentioning decreases, JNHPP, hydro or PPAs (keyword match; noisy) | FY2019/20–FY2024/25 |
| `ewura_petroleum` | total local/transit imports; imports by product (incl. HFO); consumption by product (litres). `petroleum_imports_estimated`: HFO local imports 2016–2018 estimated from stated % changes (see below) | totals 2015–FY2022/23; imports by product 2012–2015 (appendix annual totals) and 2019–FY2024/25; consumption 2014–FY2023/24 |
| `ewura_petroleum_narrative` | sentences mentioning HFO, fuel oil or power generation | 2014–FY2024/25 |
| `ewura_cap_prices` | monthly retail and wholesale cap prices (TZS/litre) at Dar es Salaam, Tanga and Mtwara; 2022 subsidy and floor prices labelled in `x_price_basis` | 2017, 2021–Oct 2026 (with gaps) |

### Bank of Tanzania datasets

| File | Contents | Coverage |
|---|---|---|
| `bot_gas` | gas production (by field, or zone total in older reports) and consumption by sector (power, industries, vehicles, households, others), MMscf. The data originate from TPDC, as stated under each table (`x_original_source`) | quarterly and FY 2013/14 – Mar 2026; by sector from FY2019/20 |
| `bot_electricity` | generation by zone; by source (TANESCO/private × hydro/thermal/solar) from FY2019/20; **by plant (TANESCO, IPTL, Songas, Symbion, Aggreko, hydro plants) for 2015–2017**. Data from TANESCO | FY2013/14 – Mar 2026 |
| `bot_trade` | imports of petroleum products, fuel and lubricants, and total goods, in USD million, cumulative over the 12 months ending in the stated month | monthly, Sep 2018 – Aug 2026 (with gaps) |
| `bot_exchange_rate` | TZS/USD annual average and end of period | 2010–2025 |

BoT caveats:

* **Unit labels as printed:**
  * 2016 electricity tables say "000' KWh".
  * 2016–17 tables say "'000 MWh".
  * 2017–18 tables say "Megawatts", a power unit used for energy. These rows are flagged `unit_label_suspect_power_unit_for_energy`.
  * Some 2019 gas tables say "MSCF" for MMscf-sized values (flagged).
  * Nothing is converted.
* **"Total" generation changes definition:** it includes imports in the 2020–21 reports and excludes them from about 2023. See `x_total_definition`.
* **Thermal generation is not split** into gas vs HFO/diesel. In practice Dar es Salaam thermal is mostly gas, and the 2015–17 plant table names IPTL (HFO) separately.
* **Extraction repairs:** numbers glued together in the PDF text ("423,596.11,313,257.1") or split apart ("3 4,770.7") are repaired, and flagged `numbers_repaired_from_extraction`.
* **Skipped rows:** a row with fewer numbers than period columns (e.g. imports from a new partner country) is skipped and logged rather than guessed.
* **Not parsed:**
  * 2014–2016 MERs print their import tables rotated, so they cannot be read; petroleum import values start in Sep 2018.
  * Zonal reports Mar-16, Jun-21 and Sep-22 had no recognisable energy table.
* **Revisions (`qa_discrepancies.csv`) that look like source errors:**
  * The Dec-23 zonal report gives Dec-2022 gas-for-power as 16,150.8, identical to Sep-2022; earlier reports say 18,243.4.
  * The Dec 2025 MER restates year ending Nov 2021 petroleum imports as 1,942.5, where earlier issues say 1,807.0.

  In both cases the `latest_edition` rule picks the newer, questionable value.

### TPDC datasets (OCR)

| File | Contents | Coverage |
|---|---|---|
| `tpdc_gas` | TPDC's **own** gas volumes (not national totals), read by OCR from the audited financial statements: gas dispatched to power plants, industries, households, CNG vehicles; gas received at and exported from the Madimba / Songo Songo plants; KPI rows giving target vs actual gas processed through the national pipeline (NNGI) | FY2014/15 – FY2024/25. By-sector split only in FY2020/21+; earlier years give totals or target/actual only |

TPDC caveats:

* The statements are scanned images. Text comes from **RapidOCR**, run only on the directors' / governance report section and cached in `data/raw/_cache/ocr/`.
* Volumes are extracted from sentences. Each value keeps its `x_sentence` and `pdf_page`, carries `ocr_extracted;narrative_extraction`, and **should be spot-checked against the page image**.
* Category comes from the nearest keyword; the period is the report year unless the number follows "from" / "compared to" or is followed by an explicit FY.
* `same_value_reported_for_other_year` flags a figure printed identically for two years. Examples:
  * the actual gas processed (12,441 MMscf) is the same in FY2015/16 and FY2016/17;
  * two-column OCR in the FY2023/24 report repeats FY2022/23 processing figures.
* Units are as printed: MMscf, and Bcf in FY2018/19–FY2019/20 ("Processed and transported 15.48 Billion Cubic Feet").

### Ministry of Energy datasets

| File | Contents | Coverage |
|---|---|---|
| `nishati_narrative` | sentences on gas, HFO and power generation (English and Swahili, e.g. "gesi asilia… futi za ujazo milioni 41,530.94") from budget speeches and the Power System Master Plan, with page numbers; values are **not** parsed out | budget years 2022/23 – 2026/27 |
| `nishati_cap_prices` | EWURA cap-price notices republished on the Ministry site. The only one, June 2022, is a scanned Swahili notice: it is OCR'd, but its before/after-subsidy table layout is not parsed, so it remains a **known gap**. The OCR text is in `data/raw/_cache/ocr/` | none parsed |

### TANESCO datasets

| File | Contents | Coverage |
|---|---|---|
| `tanesco_generation` | units generated in kWh: grid own generation (hydro, thermal), grid imports (IPPs, SPPs and imports, as labelled), isolated off-grid stations, total | FY2020/21 – FY2024/25 (two years per annual report) |

TANESCO caveats:

* **Why Playwright:** the site is a JavaScript app. The public annual-report page is rendered in headless Chromium and the document list its own JavaScript loads is read. No protected endpoint is called directly. The PDFs are public and are downloaded with the normal client.
* If rendering fails, the scraper reuses the documents in earlier manifests (offline fallback).
* **Certificates:** TANESCO's server omits an intermediate certificate. Certificates are still verified, using the Windows certificate store (`truststore`).
* **Known source error:** the 2024/25 report prints FY2024/25 grid imports as 83,743,720.42 kWh. The rows then fall exactly 700,000,000 short of the printed total, which suggests a dropped leading 7. The total is flagged; nothing is corrected.
* The 2017-18 annual report has no such table.

### News datasets

| File | Contents | Coverage |
|---|---|---|
| `news_dailynews` | Daily News full articles from its public WordPress API, kept only if the title or text contains an exact topic phrase. Columns: `x_topics` (natural_gas, hfo, pipeline, gasco, tpdc, songo_songo, mtwara_mnazi, jnhpp, gas_power_plants), `x_text`, `x_word_count`. **Full text is in the CSV only**; the .dta truncates text to 2,000 characters | Aug 2021 – present (≈1,200 articles). The API holds nothing older: **news for 2015 – mid-2021 is a gap** |
| `news_citizen` | The Citizen headline, date and URL only (no article text: paywalled, and site search is disallowed by robots.txt) | builds up per run. The sitemap covers only the last 1–2 days, so run `python -m pipeline run --source citizen` daily |

Topic matching: "Mtwara" or "pipeline" alone is not enough; a gas/energy topic must also match. Definitions are in `parsers/news.py`.

### NBS datasets

| File | Contents | Coverage |
|---|---|---|
| `nbs_energy_balance` | every cell of the 2022 Energy Balance workbook: flow (`x_flow`, e.g. Imports, Main activity producer electricity, Industry) × product (`category`, e.g. Natural gas, Fuel oil, Gas/diesel oil) × unit, for Mainland Tanzania and the United Republic (`x_geography`), aggregated (T1/T3) and detailed (T2/T4) tables (`x_table`). Units as published: ktoe, plus TJ and GWh blocks that the workbook itself supplies (aggregated tables only). Excel `#VALUE!` cells are empty and flagged | 2022 |
| `nbs_power_generation_inputs` | the same rows filtered to power-generation flows (`x_power_generation_flow`): electricity plants (aggregated), and main-activity producers and autoproducers (detailed) | 2022 |
| `nbs_narrative` | energy sentences from the 2022 Energy Balance Key Insights and Methodology PDFs | 2022 |

Key 2022 figures (Mainland, ktoe):
* **Main-activity (grid) power plants** used 1,456.6 natural gas, **4.0 fuel oil (HFO)** and 22.1 diesel, alongside 219.7 hydro, and produced 800.0 electricity.
* **Autoproducers** used 75.7 gas and 166.3 diesel.
* The detailed rows sum exactly to the aggregated "Electricity plants" row: gas 1,532.2, oil products 192.5, electricity 902.2.

NBS publishes no other energy series on its site. The Statistical Abstract is not online in a findable form.

### PBPA (surveyed, not parsed)

* PBPA's monthly bulk-procurement tender results (2019–2026) list only planned cargoes of gasoil, petrol and jet/kerosene in tonnes, plus supplier premiums. **HFO is not procured through the bulk procurement system.**
* The "Statistical Petroleum Importations" page is empty.
* The energy-sector reports are copies of EWURA's.
* Skipped by decision (2026-10-15); it can be added later for import-cost analysis.

## Cross-source checks (`qa_cross_source.csv`)

`python -m pipeline qa` compares explicitly mapped series between EWURA and BoT
for each fiscal year, using each source's latest edition. Key result: production
agrees within 0–2 %, but **consumption differs widely**:

* Gas for power: EWURA 49,876 vs BoT 58,226 MMscf in FY2021/22, and 51,610 vs 68,788 in FY2022/23.
* EWURA's own supply exceeds its consumption by about 9,500 MMscf in FY2021/22 and 17,800 MMscf in FY2022/23.
* BoT consumption roughly equals production.

**Decision (2026-10-15): BoT/TPDC (`bot_gas`) is the main series for gas production
and consumption, and EWURA (`ewura_gas`) is kept as a robustness check.** The
`preferred_source` column in `qa_cross_source.csv` records this. No values are
altered. The `latest_edition` rule is kept even where the newest edition looks
like a source error; such cases stay flagged in `qa_discrepancies.csv`.

## Known data gaps and caveats

* **Gas consumption by customer category is not available as table text before
  FY2020/21.** The FY2019/20–FY2021/22 reports show it only as charts. So
  the power-sector share of gas from EWURA starts in FY2020/21.
* **No EWURA natural gas report before FY2019/20 is published** on the site.
* **HFO imports by product, 2012–2019:**
  * 2012–2015 are exact: annual "Total" rows of the monthly appendix tables in the
    2013/14 report (sideways pages, re-read with pypdfium2) and the 2015 report
    (a scan, OCR'd after rotating the page; `qa_flag` `ocr_extracted`). Product
    columns are checked against the printed row or grand total. The 2015 HFO total
    includes 43.8 M litres of transit HFO "localized" for the local market.
  * 2016–2018 are **estimates** (`petroleum_imports_estimated`). The 2016–2018
    reports show product splits only in charts, and the 2019 report's Appendix 9
    uses a font that cannot be decoded. The estimates chain the stated HFO
    changes (2016 −75 %, 2017 −30 % for local imports, both from 2015's exact
    value; 2018 backed out of 2019's −4 %) and are checked against the stated
    HFO shares of local imports (2 % in 2016, 1 % in 2017; `x_share_check`).
    The 2019 sentence does not say "local" explicitly, but sits in the local
    imports paragraph.
  * The exact HFO **consumption** series (sales, 2014–2019) needs no estimation and
    is the safer series for showing the post-2016 decline.
  * Consumption exceeds local imports in every year 2016–2022, including the
    exact years 2019–2022 (≈ 32 M litres over four years), so imports and consumption do not
    reconcile as a simple stock balance. The 2017 report also gives a 53 % rise
    in *transit* HFO, although the 2016 report says there were no transit HFO
    imports in 2016.
  * The 2016 report states that HFO's share of generation fell from 21 % (2015)
    to 6 % (2016). See `ewura_petroleum_narrative`.
* **Cap prices:**
  * EWURA's site has no notices for 2018–2020 or Jan–Jun 2017.
  * Six notices are present but could not be parsed (logged): Aug and Sep 2017,
    Jan, Sep and Oct 2021, and Jun 2022.
  * Cap prices cover petrol, diesel and kerosene only, not HFO.
* **Terminology:** EWURA labels the plant series "processed" (FY2019/20–20/21),
  "produced" (FY2021/22) and "supplied/exported" (FY2022/23+) across editions.
  The values match between editions, so they share one variable
  `gas_supply_by_plant`, and the label used is kept in `x_measure_as_labelled`.
  From FY2023/24, EWURA also reports field-level production (`gas_produced_by_field`),
  which is a different measurement point.
* **Inconsistencies already found in the sources (flagged, not fixed):**
  * FY2023/24 CNG consumption: the printed total is 308.72 MMscf, but the sellers sum to 518.35.
  * FY2023/24 industry consumption: the printed total is 11,639.30 MMscf, but the sellers sum to 11,232.69.
  * FY2020/21 "Households, Commercial and Institutions" is 2,267.85 MMscf (3.8 %), against 11–18 MMscf in later years.
  * FY2021/22 household connections: 1,407 in the FY2021/22 report vs 970 in later reports.
  * FY2023/24 Songas and M&P plant throughput differ between the FY2023/24 and FY2024/25 reports.
  * Supply (81,067 MMscf) vs consumption (63,273 MMscf) for FY2022/23. These are different concepts, but the gap is large.
* **Glyph decoding:** several older EWURA PDFs lack a ToUnicode map, so text
  extracts as `(cid:N)`. These glyph IDs are a constant offset from ASCII. A page
  is decoded only if the result contains known words, and every value from such a
  page carries `text_glyph_decoded`.
* **Listing dates are unreliable** (e.g. the 2021 petroleum report is dated
  2018-12-12). Periods are always taken from the document contents.

## NLP analysis: how Daily News covers natural gas

*Work in progress. Tasks 1, 2 and 5 are done: corpus preparation, topic models,
and the link to the energy series. Tone validation (Task 3) is waiting for hand
labels, and the framing classifier (Task 4) follows it.*

The analysis asks three questions:

1. Which themes dominate energy news, and how do they change over 2022–2026?
2. Did coverage move from natural gas toward hydropower (JNHPP) around the 2024 fall
   in gas consumption? The question is whether coverage *tracks or anticipates* the
   gas-to-hydro shift, not whether it explains the drop.
3. Is coverage of natural gas becoming more positive or more negative?

All code is in `nlp/` and runs in its own environment, so the scraper environment
stays light. Committed outputs contain ids, URLs, counts and model terms only.
Article text stays in `data/nlp/`, which is gitignored.

```powershell
python -m venv nlp\.venv
nlp\.venv\Scripts\python.exe -m pip install -r nlp\requirements.txt
nlp\.venv\Scripts\python.exe -m pip install -r nlp\requirements-embed.txt   # torch (CPU) + BERTopic, ~217 MB
nlp\.venv\Scripts\python.exe nlp\scripts\01_prep.py             # clean corpus, coverage tables
nlp\.venv\Scripts\python.exe nlp\scripts\01b_near_duplicates.py  # rewritten reposts (review nlp\config\near_duplicate_decisions.csv, then re-run 01)
nlp\.venv\Scripts\python.exe nlp\scripts\02_topics_nmf.py       # NMF baseline (~9 min k-sweep)
nlp\.venv\Scripts\python.exe nlp\scripts\02b_topics_bertopic.py # BERTopic + comparison
nlp\.venv\Scripts\python.exe nlp\scripts\02c_gas_mentions.py    # gas discussed vs listed
nlp\.venv\Scripts\python.exe nlp\scripts\05_link_energy.py      # coverage vs BoT gas consumption
cd nlp; .venv\Scripts\python.exe -m pytest                        # tests
```

Every random step uses a fixed seed (42), and the embedding model is pinned to a
specific Hub commit.

### Methods

* **Corpus.** The corpus is the Daily News articles from the scraper (1,225). Text is
  re-extracted from the raw HTML so that inserted "ALSO READ: <other headline>"
  links and photo captions can be dropped.
  * Topic tags are re-matched on the clean text. This removes 36 articles whose
    only gas or energy keyword was in a link to another story.
  * Duplicates are removed in two passes, keeping the earliest copy each time:
    * 15 near-identical reposts (word 5-gram Jaccard ≥ 0.8)
    * 15 rewritten reposts that word overlap misses
      (`01b_near_duplicates.py`). These are found from embedding similarity,
      shared text and publication dates, then reviewed by hand in
      `nlp/config/near_duplicate_decisions.csv`:
      * 14 reposts with ≥ 70% shared text, sometimes reprinted months later
      * 2 paraphrased rewrites, published within a week
      * 22 pairs of partial text reuse, which are kept as separate articles
  * That leaves 1,159 articles. The **main analysis window starts in 2022-Q3**
    (1,157 articles), because the API returns almost nothing earlier. Every quarter
    from 2022-Q3 has at least 34 articles. Months with fewer than 10 articles, and
    the partial current month, are flagged in `nlp/tables/prep_articles_per_month.csv`.
* **NMF baseline.** NMF runs on TF-IDF (unigrams and bigrams).
  * Officials' names are stop words, defined as words that follow a title (Mr,
    Minister, …) in at least 5 articles and are capitalised in at least 80% of
    their uses. Without this, the model builds one topic per minister, and those
    topics follow cabinet reshuffles rather than themes.
  * **k = 17 is fixed, not chosen by the k-sweep.** Over k = 10–25, coherence is
    flat, and the stability score varies as much between two runs (SD ≈ 0.02) as
    it does across k. The sweep (`nmf_k_selection.csv`) is therefore a diagnostic,
    and key results are repeated at k = 14, 15, 17, 19 and 21
    (`nmf_k_sensitivity_summary.csv`).
  * Topic shares by quarter come with article-bootstrap 95% intervals. They are
    repeated without the 30% of articles that mention a keyword only once, as a
    robustness check.
* **BERTopic.** BERTopic uses all-MiniLM-L6-v2 on CPU, with UMAP/HDBSCAN
  (`min_cluster_size = 15`).
  * The model reads at most about 200 words, so articles are embedded in 150-word
    chunks, and the chunk vectors are averaged.
  * The comparison with NMF is in `topic_model_comparison.csv`:

  | | NMF (k=17) | BERTopic |
  |---|---|---|
  | Topics | 17 | 19, plus 18% outliers |
  | Coherence (NPMI, same reference vocabulary) | 0.42 | 0.34 |
  | Topic diversity | 0.96 | 0.94 |
  | Stability (AMI of assignments across 5 seeds) | 0.73 | 0.77 |
  | Agreement between the models (AMI) | 0.59 | |

  * Removing the 15 rewritten reposts changed 2 of the 17 NMF topics.
    * "Zanzibar–Union affairs" disappeared, because it rested on duplicated
      copies of a few Union articles.
    * "Local communities & CSR", the least coherent topic, also disappeared.
    * "EWURA & utility regulation" and "China relations" took their places.
    * The other 15 topics kept 11–15 of their top 15 terms.

* **How gas is mentioned.** A rule-based classifier (`energy_nlp/mentions.py`,
  rule version 2, following codebook v4) labels each gas mention as one of:
  * *discussed*: a sentence says something about gas itself
  * *in an energy-mix list*: "hydropower, natural gas and solar" (rule T1b)
  * *in a project or sector list*: "projects in energy, natural gas and roads"
    (rule T1c)
  * *in a resource list*: "minerals, natural gas and wildlife" (rule T1)

  In the codebook, `energy_mix_list` covers **both** energy-mix lists and
  project/sector lists (T1c). The coder marks the project/sector rows `proj_list`
  in the notes, so results can be reported with and without them. "Oil and
  natural gas sector" is a sector name, not a mention of gas (T4a).

  The rule was built from the codebook and from articles outside the labelled
  sample, not from the labels. **It is indicative until validated** against the
  hand labels.

### Findings so far (descriptive)

* **Gas is mentioned more but discussed less.**
  * The share of articles that mention natural gas rose from **52% (2022-Q3 to
    2024-Q4) to 62% (2025-Q1 to 2026-Q3)**.
  * Among gas-mentioning articles, the share that actually discuss gas fell from
    **84% to 73%** (−10.7 points, 95% CI −17.2 to −3.9). This uses the rule-based
    classifier, so it is indicative until validated.
  * **About 81% of the rise** (8.2 of 10.1 points) is articles where gas appears
    only in lists:
    * energy-mix lists +4.5 points [1.9, 7.3]
    * resource-wealth lists +2.8 [0.8, 4.8]
    * project/sector lists +1.0 [−1.0, 3.0]

    Without the project/sector lists, list-only mentions still account for 72% of
    the rise. The rise in articles that discuss gas, +1.9 [−3.7, 8.3], cannot be
    told apart from zero.
  * Adding project/sector lists to the rule (codebook v4) **strengthens** this
    finding. Under the earlier rule (v1), the drop was 90% → 81% (−9.4 points),
    and lists accounted for about two-thirds of the rise.
* **The rise is not gas-sector news.** In both models the gas-sector topics stay flat
  or fall: LNG, CNG and gas fields. The increase comes from general economic and
  development coverage.
  * In NMF, the industrialisation and regional growth topic adds +5.0 to +7.3
    points at every k tested.
  * In BERTopic, the rise is spread across several clusters: the broad power and
    renewables cluster (+3.3 points), outliers, Middle East fuel-price shocks, the
    oil-and-gas sector cluster, economic diplomacy and macroeconomic coverage. In
    both models the LNG topic falls (BERTopic −2.3 points, 95% CI −4.0 to −0.7).
  * The two models disagree on clean cooking (NMF +2.8 points, BERTopic −1.1),
    because NMF's topic is broader.
  * Without passing mentions, all 17 NMF topic trends keep their direction.
* **Hydropower.** The JNHPP dam topic is largest in 2022–23, while the dam was being
  built. It falls by about 6 points of coverage between the two periods. The first
  turbine (2024-Q1) shows up as a one-quarter jump in articles tagged JNHPP.
* **Coverage and the gas-to-hydro shift** (`link_*.csv`, `link_gas_jnhpp.png`).
  This compares quarterly coverage with BoT/TPDC quarterly gas consumption
  (latest editions). It is descriptive only.
  * **The 2024 drop.** GASCO throughput (author's data) fell **14.5%** between
    calendar years 2023 and 2024, from 49,148 to about 42,017 MMscf. BoT/TPDC
    data support this, showing a fall of 19.0% in total gas consumption and
    23.1% in gas for power over the same calendar years. The two sources differ
    in coverage and definitions, so their percentages are not expected to match.
  * **Timing.** Gas for power first falls below its 2023 range in 2024-Q2, and
    the largest quarterly fall is in 2024-Q3. This follows the first JNHPP
    turbines: turbine 9 in February 2024, and three turbines (705 MW) by July
    2024. Milestone dates come from Daily News reports (`jnhpp_milestones.csv`).
  * **Coverage *tracks* JNHPP events rather than gas use.**
    * Articles focused on JNHPP peak at reservoir impounding (2022-Q4), six
      quarters before the drop.
    * Articles mentioning JNHPP peak in 2024-Q1, when the first turbine came on.
    * Gas coverage shows no matching fall: the share of articles mentioning gas
      dips in 2024-Q1, then rises.
  * **No lead/lag relation between coverage and gas use.** The analysis
    correlates quarter-on-quarter changes in coverage with changes in gas for
    power, at leads and lags up to ±3 quarters (about 11–14 quarters each).
    * Only 1 of 28 correlations passes the 5% threshold, which is about what
      chance alone would produce.
    * Its sign is also the opposite of "coverage anticipates the drop".
    * The coverage series therefore neither anticipates nor explains the fall in
      gas use. Both follow the physical arrival of JNHPP capacity.
  * **Data note.** The latest edition repeats the Sep-2022 value for Dec-2022.
    That value is kept as published and flagged, and results are repeated
    without it. 2022-Q2 is not reported.

### Limitations

* **Time span.** The API holds nothing before mid-2021 and very little before
  2022-Q3, so the 2015–2021 period of the energy data has no news counterpart.
* **One newspaper, government-owned.** Daily News is state-owned (TSN), so its
  coverage reflects official framing and cannot stand for Tanzanian media in
  general. *The Citizen* is available as headlines only, and Swahili outlets are
  not covered.
* **Keyword-defined corpus.** Articles enter the corpus by keyword, including the
  JNHPP keywords. Trends are shares *within* energy-keyword coverage, not shares of
  all news.
* **Model choices.** The number of NMF topics cannot be identified from the data
  (k is fixed and sensitivity-tested). BERTopic leaves 20% of articles as outliers.
  Topic labels are the author's.
* **Descriptive only.** None of these results identify causal effects of coverage
  on gas use. The quarterly overlap with the gas data is short (2022-Q3 to
  2026-Q1), so the lead/lag analysis has little statistical power.
