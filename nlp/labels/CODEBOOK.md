# Coding manual: tone toward natural gas, gas mention type, and primary frame

Version 3 (approved), October 2026.

There is one coder, the author. To measure intra-coder reliability (Cohen's kappa),
30 of the 150 articles are coded a second time, at least 14 days after the first
round.

## Unit and procedure

- **Unit:** one article (headline plus full body text).
- Read the headline and the whole article before coding.
- Code what the text says, including quoted sources such as ministers, officials and
  companies. Don't discount a statement because of who made it or whether you
  believe it. Outlet bias is analysed separately, not corrected for during coding.
- Code three variables, in this order: `gas_mention`, then `gas_tone`, then `frame`.
- Tick `unsure` if you hesitated between two codes on any variable. In `notes`,
  write which variable and which two codes, e.g. "frame: economic_opportunity /
  governance_investment".

## What counts as natural gas

Natural gas means **methane gas and its uses**: gas fields and wells, processing
plants, gas pipelines, gas-fired power, piped household or industrial gas, CNG
(compressed natural gas) and LNG (liquefied natural gas).

These are **not** natural gas:

- LPG cooking-gas cylinders
- crude oil and EACOP
- petrol, diesel and HFO
- hydropower

An article that mentions only these is coded `gas_mention = none` and
`gas_tone = not_about_gas`.

## Variable 1: `gas_mention`, how the article mentions natural gas

| Code | Definition | Example |
|---|---|---|
| `substantive` | At least one full sentence says something about gas itself: its supply, price, use, project progress or effects (rule T4). | "Natural gas production at Madimba more than doubled in 2023." |
| `resource_list` | Gas appears **only** as an item in a list of the country's natural resources or assets (rule T1). | "endowed with minerals, natural gas, wildlife and fertile land" |
| `energy_mix_list` | Gas appears **only** as an item in a list of energy sources or technologies (rule T1b). | "investing in hydropower, natural gas and renewable energy" |
| `none` | No natural gas mention, e.g. only LPG, oil or hydropower. | an EACOP article mentioning only crude oil |

**Precedence when an article has several mentions:**

1. If any mention is substantive, code `substantive`.
2. Otherwise, if any is in a resource list, code `resource_list`.
3. Otherwise, code `energy_mix_list`.

"Oil and gas", as in "the oil and gas sector", is a name for a sector, not a list.
Judge the rest of that sentence.

## Variable 2: `gas_tone`, the tone toward natural gas

| Code | Definition | Example |
|---|---|---|
| `positive` | Gas is presented mainly as a benefit, success, progress or opportunity, **including factual reports of growth or success** (rule T2a). | "Madimba production more than doubled"; "CNG saves motorists 40%"; "gas-fired plant adds 185 MW to the grid" |
| `neutral` | Gas is discussed substantively but the content is **procedural, with no outcome**: tenders, meetings, bare announcements, statistics without evaluation or change, or gas described as being replaced without an unfavourable judgement (T5). | "TPDC invites bids for the Lindi gas distribution network"; "Tanzania's gas reserves stand at 57 tcf"; "JNHPP will reduce reliance on gas-fired plants" |
| `negative` | Gas is presented mainly as a problem, failure, risk, cost, delay or conflict, **including factual reports of failure or decline** (T2a) and unfavourable comparisons with other sources (T5). | "LNG talks stall again"; "gas output fell 20% after a plant shutdown"; "gas power costs twice as much as hydro" |
| `not_about_gas` | The article says nothing evaluative about gas itself. This is always the code when `gas_mention` is not `substantive`. | one line, "...projects in energy, natural gas and roads..." |

**Rules**

- **T1, resource-list (endowment) rule.** If gas appears only as an item in a list
  of the country's resources or assets, code tone `not_about_gas`, `gas_mention`
  `resource_list` and frame `resource_wealth`.
- **T1b, energy-mix-list rule.** If gas appears only as an item in a list of energy
  sources or technologies, code tone `not_about_gas` and `gas_mention`
  `energy_mix_list`. The frame follows the article's energy content (see
  Variable 3).
- **T2, mixed tone.** Code the balance of the article. When it's genuinely even,
  let the headline and the first three paragraphs decide. If those are even too,
  code `neutral`.
- **T2a, factual outcomes.** A factual report of growth, success or progress
  ("production doubled", "connections reached 1,500 households") is `positive`. A
  factual report of failure or decline is `negative`. `neutral` is only for
  procedural content with no outcome: tenders, meetings, bare announcements, and
  statistics that report a level without a change or evaluation.
- **T3, plans and promises.** Code a plan or announcement as `positive` only if the
  text presents it as beneficial ("will transform the region"). A bare announcement
  of a plan is `neutral`.
- **T4, substantive vs. passing.** Gas counts as more than a passing mention if at
  least one full sentence says something about gas itself: its supply, price, use,
  project progress or effects. Only `substantive` articles can be `positive`,
  `neutral` or `negative`.
- **T5, comparison and displacement.** If gas is contrasted unfavourably with
  another source (costlier, dirtier, less reliable), code `negative`. If gas is only
  described as being replaced or reduced, e.g. by JNHPP, without an unfavourable
  judgement, code `neutral`.

## Variable 3: `frame`, one primary frame per article

The frame is **how the article presents gas**. If `gas_mention` is not
`substantive`, code the frame of the article's energy content instead. Rule T1
overrides this: a `resource_list` article is always `resource_wealth`.

| Code | Frame | Definition | Example cue |
|---|---|---|---|
| `economic_opportunity` | Economic opportunity & industrialisation | Gas or energy as a driver of growth, investment, jobs, industry or revenue: the *outcomes*. | "will attract investors and create 10,000 jobs"; "gas to feed a fertiliser plant" |
| `supply_reliability` | Supply & reliability | Availability and performance: generation, capacity, outages, gas supply volumes, infrastructure working or failing. | "added 150 MW to the national grid"; "power cuts in Dar" |
| `prices_costs` | Prices & costs | Prices, tariffs, subsidies and costs or savings for users or the state. | "EWURA caps pump prices"; "CNG is cheaper than petrol" |
| `local_communities` | Local communities & compensation | Effects on host or local communities: benefits, grievances, compensation, CSR, local content, resettlement. | "Mtwara residents demand benefits"; "compensation paid to villagers" |
| `environment_climate` | Environment & climate | Emissions, deforestation, clean cooking, energy transition, environmental risk. | "cut charcoal use"; "cleaner fuel" |
| `governance_investment` | Governance & investment deals | The *process*: contracts, negotiations (e.g. the LNG host government agreement), licensing rounds, regulation, state enterprises, accountability. | "HGA negotiations concluded"; "PURA licensing round" |
| `resource_wealth` | National resource wealth / endowment | Gas appears mainly as part of the nation's resource wealth or natural assets, not as a sector with its own story. | "endowed with vast mineral deposits, natural gas reserves…" |
| `other` | Other | None of the above fits, e.g. accidents, court cases, sports sponsorship, obituaries. | "TPDC staff win football bonanza" |

**Rules**

- **F1, one frame.** If several frames appear, choose the one in the headline and
  lead (the first three paragraphs). If that doesn't settle it, choose the one that
  takes up the most text.
- **F2, economic vs. governance.** Deals, contracts, negotiations and regulation are
  `governance_investment`. Growth, jobs and revenue as results are
  `economic_opportunity`.
- **F3, endowment vs. economic.** Use `resource_wealth` when gas is a static asset
  in a list. Use `economic_opportunity` when the text links gas to a specific use or
  outcome, such as a fertiliser plant or industrial gas supply.

## Consistency checks (run by the scripts, not by the coder)

- `gas_mention` ≠ `substantive` ⇒ `gas_tone` = `not_about_gas`
- `gas_mention` = `resource_list` ⇒ `frame` = `resource_wealth`
- `gas_mention` = `none` ⇒ `gas_tone` = `not_about_gas`

Rows that break these rules are listed for you to fix. They are never corrected
automatically.

## Sample (stratified)

The sample is 150 articles from the 1,172 main-window articles, stratified by
**year** (2022 to 2026) × **gas-tagged / not gas-tagged** (the scraper's
`natural_gas` tag). Each year gets 20 gas-tagged and 10 other articles. Gas articles
are oversampled because RQ3 concerns tone toward gas. Each article carries a
sampling weight w = N_h / n_h (articles in its stratum ÷ articles sampled from it),
so estimates can be re-weighted to the full corpus. The draw uses seed 42. Sheet
order is shuffled so that year and stratum are not visible from the row position.

## Files

- `data/nlp/labelling_sheet.xlsx` (**not committed**, it holds article text):
  a *Labels* tab with `sample_id, title, text`, plus blank columns `gas_mention,
  gas_tone, frame, unsure, notes`, where the codes and `unsure` (`x`) are
  dropdowns that accept only valid values. A *Codebook* tab summarises this
  manual. For blind coding, the post id, publication date, URL, stratum
  and weight are kept only in `sample_design.csv`. Dates mentioned *inside* the
  text cannot be hidden.
- `nlp/labels/sample_design.csv` (committed): `sample_id, post_id, pub_date, url,
  year, gas_tagged, stratum, N_h, n_h, weight`.
- `nlp/labels/labels_round1.csv` (committed, **no text**): `sample_id, post_id,
  gas_mention, gas_tone, frame, unsure`, written from the completed sheet by the
  script.
- Re-coding: 30 articles drawn at random from the 150 (seed 43), shuffled, with
  codes blank, to be coded at least 14 days after round 1.
