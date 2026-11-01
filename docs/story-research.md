# PerfPilot — Story Research: How to Tell This Project for Judging

> Scope: primary/high-trust sources only. Every material claim below cites a page that was fetched and read in full (not a search snippet). Local project facts are from [PLAN.md](../PLAN.md) §5–6 and [README.md](../README.md) as read 2026-11-01.
>
> PerfPilot in one line (from README): an agent that keeps a vLLM deployment of a Meta open model fast — **PerfBisect** finds the first slow commit, **TuneSmith** recovers speed via serve flags, and every claim is benchmark-verified with intervals.

---

## 1. What judges actually reward (per primary sources)

### 1.1 This hackathon's own rubric — read this first

The Meta Global AI Developer Hackathon (Cerebral Valley × Meta) is the governing rubric. Two layers matter:

**Track lens (Indie — PerfPilot's track).** The official event page states the Indie track is *"Judged on creativity, technical ambition, and whether it works,"* while the Startup track is judged on technical execution / production-readiness. PerfPilot is Indie, "Push the Frontier" theme. Source: [Meta Global AI Developer Hackathon — Cerebral Valley](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) and the mirror at [dev.meta.ai/events/global-hackathon](https://dev.meta.ai/events/global-hackathon) (both fetched; wording identical).

**Weighted score (from the official rules as reported in the rules mirror).** Finished projects are scored on **impact (30%), how creatively they use Meta's AI tools (25%), the demo including its video (25%), and technical depth and execution (20%)**. A panel chosen by the organisers picks finalists. Source: [Meta is offering $1 million in prizes… — madrobot.blog rules summary (citing the official rules published on the Cerebral Valley page)](https://madrobot.blog/2026/10/03/meta-global-ai-developer-hackathon-1-million-prizes-muse-model-api/). Treat the Cerebral Valley page as authoritative if numbers ever conflict; the mirror is the only fetched source that prints the weights.

**Theme fit.** Both official pages frame the two themes identically: *"Build for Your World"* (a problem you know firsthand) vs *"Push the Frontier"* (software/robotics/manufacturing/energy/materials — *"Give a machine a new capability, accelerate a scientific experiment, or unlock a better way to understand and control a physical system"*). Sources: [Cerebral Valley event page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon); [dev.meta.ai/events/global-hackathon](https://dev.meta.ai/events/global-hackathon). PerfPilot's "agent that finds and fixes vLLM regressions" is a textbook Push-the-Frontier fit — say so explicitly in the first 15 seconds and in the submission's first paragraph. Judges score theme alignment whether or not it is a separate line item (see §1.2).

**Logistics judges will check before creativity.** Submissions due Nov 4 9 PM ET; judging Nov 4–8; live finals Nov 9. Sources: [Cerebral Valley event page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon); [madrobot.blog rules summary](https://madrobot.blog/2026/10/03/meta-global-ai-developer-hackathon-1-million-prizes-muse-model-api/). PLAN.md §5 already encodes the Wed-noon-Phoenix internal deadline — keep it.

### 1.2 The generic Devpost rubric (what most judges carry in their heads)

Devpost's own guidance — the platform most Cerebral Valley judges have used before — says organisers set their own criteria, but submissions are *"most often scored against: Technological implementation, Ease of use, Demonstration (video), Potential impact, Quality of the idea, Design."* Source: [Understanding hackathon submission and judging criteria — Devpost](https://info.devpost.com/blog/understanding-hackathon-submission-and-judging-criteria).

What each means, in Devpost's words (paraphrased tightly, source same as above):

- **Technological implementation** — how well you used the required tools, plus complexity/sophistication. *"Don't be afraid to push boundaries and showcase your technical expertise."*
- **Ease of use** — intuitive, smooth UX. Test: *"If you can't explain how to use your project in a sentence or two, it might be too complicated."*
- **Demonstration** — *"Your demo video is your chance to shine. It should clearly and concisely explain your project, its features, and its benefits. A well-crafted demo can significantly boost your chances of winning."*
- **Potential impact** — does it solve a genuine problem with positive impact? *"Research your problem area thoroughly"* to articulate it.
- **Quality of the idea** — originality/innovation highly valued.
- **Design** — clean, functional, professional presentation.

**Implication for PerfPilot:** you are strong on Technological implementation and Quality of idea, weaker-by-default on Ease of use and Design (a CLI + HTML reports). The video and README must do the ease-of-use work: one-sentence usage (`perfpilot bisect --good G --bad B`), one screen that a non-vLLM judge understands (the tok/s cliff), and a polished report.html as the "design" surface. Do not let the submission read as backend-only — see the judge warning in §1.3.

### 1.3 What five seasoned judges say they actually do

Devpost interviewed five experienced judges (Square, Databricks, Google, NEAR, Atlassian). Their self-reported process is the most direct evidence available for "how judging actually happens." Source for all quotes in this subsection: [How to win a hackathon: Advice from 5 seasoned judges — Devpost](https://info.devpost.com/blog/hackathon-judging-tips).

- **Video first, text second, code third.** *"The demo video gives the most amount of scope… the video becomes crucial in giving us our first indicator of how much time was invested"* (Richard Moot, Square). *"For me personally, I watch the video to get context and then I use that for testing the submission."* Another judge: *"Presentation and storytelling matters and it can be a huge part of your success"* (Karen Bajza-Terlouw, Databricks).
- **Requirements are a filter, not a tiebreaker.** *"The first thing I do is go over the requirements… it was surprising to see how many submissions did not fulfill the basic requirements"* (Karen). *"Definitely err on more details, and always double-check you're hitting every single requirement."* → PerfPilot's bonus eligibility (open weights, agentic loop) must be named verbatim in video voiceover *and* submission text, not just implied.
- **Balance across criteria beats maxing one.** *"A project that really stands out is one that was clearly considering the judging criteria… sometimes people pick only one of the judging criteria but miss everything else"* (Richard). Negative version: *"when they over-indexed on one particular criteria at the detriment of the rest… extremely back-end heavy and has almost no front end, there's no UI"* (Richard). → Give each rubric item one visible beat (see §5 outline mapping).
- **Show, don't claim, the work.** *"Ambiguity is a red flag, and also projects that lack detail and code… video intro looked really cool… but then when you dug into the project or their GitHub, it was a lot lighter on code"* (Karen). *"The more open you can be with how you created your project, the better… demo their code… voiceover while they showed the different components."* → Show `events.jsonl`, the proposal accept/reject log, and a real diff on screen; link every claim to a `runs/<id>/` artifact.
- **Problem-behind-the-solution beats solution-looking-for-a-problem.** *"Do they see the problem behind the solution, or are they just trying to fit their solution to this particular hackathon? We have awarded great projects that were not great from the technical side… but we saw potential in the team — expertise and background knowledge on this particular problem"* (Maria Yarotska, NEAR). Red flag: same project relabelled across hackathons. → Lead with the vLLM regression pain (real upstream issues, §3.1) before naming PerfBisect/TuneSmith.
- **Finished, usable, differentiated.** *"Is that finished product something that I would want to use?… catches your eye, has something unique, differentiates it"* (Warren Marusiak, Atlassian). *"Rehashed ideas… extremely simple… just to get a submission out"* is the negative. → End on a runnable command + re-renderable report, not a roadmap slide.

### 1.4 What past winners' announcements reveal (pattern extraction)

These are official winner announcements linked from PLAN.md §6. They show the *shape* of a winning story, not just the rubric.

**LlamaCon Hackathon (Meta, 44 projects → top 6, two judging rounds).** Winners were summarised in 3–5 sentences each: problem → mechanism → demo artifact → stack. E.g. OrgLens (1st): expert-matching from Jira/GitHub/docs/resumes → knowledge graph + digital-twin chat → React/Tailwind/Django demo on GitHub + Llama API. Source: [Meet the winners of our first-ever LlamaCon Hackathon — ai.meta.com](https://ai.meta.com/blog/llamacon-hackathon/). **Pattern:** every winner summary names (a) a concrete user pain, (b) the data the system touches, (c) a clickable demo/code link. PerfPilot's equivalent: pain = "a deploy silently cost tok/s," data = commit range + benchmark trials, artifact = `report.html` + `events.jsonl`.

**Built with Opus 4.7 (Cerebral Valley × Anthropic).** Winner write-ups foreground a *personal* problem and a *measurable* before/after. Two reusable patterns:

- *MaestrIA* ("Keep Thinking" prize): a JSON knowledge file from interviews lifted eval accuracy *"seven points (74% to 81% against a human master's judgment)"* — a single honest number carrying the whole story. Source: [Meet the winners of the Built with Opus 4.7 hackathon — claude.com](https://claude.com/resources/articles/meet-the-winners-of-built-with-opus-4-7-claude-code-hackathon).
- *Virtual Puppet Theater* (Most Creative Use): the winner's own advice — *"plan time to create the demo video… It takes way longer than you think to produce a 3-minute video… reserve that entire last day just for producing the demo."* Source: same page as above. PLAN.md's "never live-GPU, record runs" rule is the PerfPilot version of this.

**Built with Opus 4.6.** Same structure: winners introduced by who they are (cardiologist, electronic musician) before what they built — identity as credibility for the problem. Source: [Meet the winners of our Built with Opus 4.6 hackathon — claude.com](https://claude.com/blog/meet-the-winners-of-our-built-with-opus-4-6-claude-code-hackathon). **Pattern for a solo/indie PerfPilot story:** one line of "I run vLLM / I got paged by a silent regression" outperforms a feature list. If that line isn't true, don't invent it — use the upstream-issue evidence instead (§3.1).

**Cross-winner synthesis (all three announcements):**

| Element | Present in winners | PerfPilot equivalent |
|---|---|---|
| Named human pain in sentence 1 | Yes | "Last deploy cost 32% tok/s and nobody knew which commit" |
| System touches real artefacts (not a toy) | Yes (Jira/GitHub, photos, webcam) | Real commit ranges, A100 trials, serve flags |
| One headline number with a method | Yes (74%→81%) | Must be an interval, not a point (see §3) |
| Demo link + code link in the summary | Yes | `report.html` (re-renderable) + repo + `events.jsonl` |
| "Built during the event" explicit | Yes | README "Built during" section (already present) |

### 1.5 PerfPilot's current story vs the rubric (gap check)

Current state (README + PLAN.md §5 as read):

- ✅ Impact line exists ("Agent that keeps a vLLM deployment… fast"), eval tables present, cost stated, open-weights/agentic eligibility named, recorded-runs-not-live-GPU rule in place.
- ⚠️ PLAN.md §5 video beats promise a *"verified uplift number"* at 0:55 (TuneSmith). The actual tune run was an honest negative (no flag beat baseline +2%; see §3). The outline in §5 below rewrites that beat — do not record the old line.
- ⚠️ The local fake-transport eval table (2/2, 0 wrong) and the GPU eval table tell different stories (see §3.2). Leading with only the clean table would violate the honesty framing that is PerfPilot's differentiator.
- ⚠️ No explicit Push-the-Frontier sentence, no one-sentence usage line for non-expert judges, no "what it costs to run" in the first screenful. All cheap to add (see §5.3).

---

## 2. Demo-video narrative structure (3 minutes)

### 2.1 What the primary guidance says

Devpost's demo-video guidance (tips collected from winning participants + Devpost staff) is the closest thing to official instructions. Source for this subsection: [6 Tips for making a winning hackathon demo video — Devpost](https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video), cross-checked against [Understanding hackathon submission and judging criteria](https://info.devpost.com/blog/understanding-hackathon-submission-and-judging-criteria) and [Advice from 5 seasoned judges](https://info.devpost.com/blog/hackathon-judging-tips).

- **Length:** *"Most hackathons require demo videos to be less than three minutes"*; *"in around three minutes"* to showcase the project; *"get as much as possible covered in the first two to three minutes with key info upfront."* Plan for **2:45–2:55**, never 3:00+ (platform rules vary; the event's own cap governs if stricter).
- **Script first, record second.** *"Write a script!… Don't be afraid to spend some extra time on your script"*; divide the time budget by section *before* recording (intro / idea / components / demo / close). Use the submission form's elevator pitch + project story as the script draft.
- **Open with the pitch, not the stack.** *"Start the video with your elevator pitch. Explain what your app does and how it addresses the hackathon in the first few seconds"*; *"Start with your pitch and a quick overview before diving in."* Judges *"will likely review multiple projects back to back, so make sure to start your video with a quick overview."*
- **Content spine:** *"focus on the most important information — what your project does, how it solves the problem, and how it works"*; answer *"What is the idea?… tell a story… What is your solution?… What tech stack did you use?"* and *"Tie it all back to your story — what problem you solved and who your project positively impacts."*
- **Show it running.** *"Make sure it's a demo! Show your project or app in action"*; *"highly visual, and interactive (show your project in use)… Spend a little time showing the technical aspects… but don't get too bogged down."* For PerfPilot, "in action" = recorded `report.html` + `events.jsonl`, never a live GPU wait (PLAN.md §5 rule; also protects against the capacity risk in PLAN.md's risk list).
- **Address the criteria out loud.** *"Address how it meets the judging criteria — are you addressing what the site actually asked for? What is the future potential?"* → Name the theme, the open-weights models, and the agentic loop verbatim.
- **Production:** use tools you already know; best microphone on the team records; upload early (YouTube unlisted/public link ready before the deadline rush); **check the video is not private** — *"if it's private, the judges won't be able to access it"*; mark "Not for Kids" on YouTube to avoid access quirks. Leave *"at least two or three hours before the end… to polish, record, and upload"* — winners say a full day (see §1.4).

### 2.2 Recommended 3-minute structure (generic, evidence-based)

Timings synthesise Devpost's "key info upfront / first 2–3 minutes" rule with the hook→problem→proof→numbers→close spine. Total 180s; target voiceover ≈ 380–420 words at natural pace.

| Beat | Time | % | Job | Evidence anchor |
|---|---|---|---|---|
| **Hook** | 0:00–0:15 | 8% | One painful, specific outcome + project name. No stack, no agenda. | "Key info upfront"; elevator pitch in first seconds |
| **Problem** | 0:15–0:40 | 14% | Who hurts, how often, why current tools miss it. One external proof point. | "Tell a story to get the audience to understand your problem" |
| **Solution in one sentence** | 0:40–0:55 | 8% | What it does + how (agent proposes, benchmarks dispose). Name theme + sponsor-tech fit. | "What it does, how it solves the problem, and how it works" |
| **Proof (recorded run)** | 0:55–2:00 | 36% | The product working: narrowing, verdict, diff, proposal log. Largest block. | "Show your project in action"; judges weight video most |
| **Numbers** | 2:00–2:35 | 19% | Eval table, intervals, cost, and the honest negative (see §3). | Technological implementation + impact scoring |
| **Close** | 2:35–3:00 | 14% | Restate impact, eligibility sentence, repo/report links, thank-you. | "Say thank you!"; "future potential" |

Rules that cut across beats:

- First frame is the result (cliff chart / verdict card), not a title slide. Judges decide in seconds whether to keep watching (see §1.3, video-first process).
- Every number on screen also spoken, with its interval or denominator ("2 of 2, zero wrong" / "−0.8%, inside noise").
- Cut every beat that doesn't map to a rubric line (impact / creative Meta-AI use / demo / technical depth). If a shot is "cool architecture," it must also answer "so what changes for the user."
- End card (last 3s, static): repo URL, "reports re-render offline: `perfpilot report runs/<id>`," X handle. Devpost tip: supplemental depth goes in text, not in a longer video.

### 2.3 What to avoid (from the same sources)

- ❌ Slides-only / deck narration (*"no slide shows/decks"*).
- ❌ Speed-ramped narration to cram features (*"Don't try to pack in way too much information or speed up the audio"*).
- ❌ Live waits (GPU spin-up, model load) — record and cut; PLAN.md already mandates this.
- ❌ Private/unlisted-broken links, last-minute uploads.
- ❌ A video that contradicts the README (e.g. promising an uplift the tune run didn't find). Judge red flag: ambiguity / video-vs-code mismatch (§1.3).

---

## 3. Framing honest negative / mixed results as rigor

PerfPilot's GPU story is mixed by design and must be told as a feature. Facts (from README, as read):

- **Baseline-outlier bug (found & fixed):** strong-overlay trial returned baseline samples `[3569, 3594, 1703]` tok/s; one collapsed run poisoned variance so nothing separated (steps=0, 10 extra runs, $3.32). Fix: `baseline_max_spread` (default 0.30) re-measures endpoints once and keeps the tighter set; proven by `test_contaminated_baseline_remeasured_finds_culprit` replaying the exact evidence (fails with the check disabled).
- **Failed eval premise:** mild overlay (`gpu-mem-0.5`, expected ~10% drop) measured −0.8% (3522 vs 3495 tok/s) — no measurable effect, so `no_regression` was the *correct* verdict on a wrong premise. README's own ≥10% keep rule says the overlay should be dropped/replaced.
- **Honest negative tune:** 6 candidates on A100, five with no interval fully above +2%, fp8 impossible on A100 (no Hopper FP8) recorded as a rejection; baseline kept, quality (50Q gate) held. *"This workload on this commit is already at its flag optimum."*
- **Guardrail that held everywhere:** 0 wrong verdicts in both GPU trials (intermediates ambiguous/skipped, endpoints labelled correctly); local protocol 2/2 culprits, 0 wrong.

### 3.1 Why honesty scores (primary-source backing)

- **Claims must match results, with limits stated.** The NeurIPS Paper Checklist — the ML community's high-trust reporting standard — requires that *"main claims… accurately reflect the paper's contributions and scope,"* that limitations/assumptions be stated, and explicitly notes: *"answering 'no'… is not grounds for rejection"* when justified. It asks for a separate Limitations section covering strong assumptions, robustness, and scope (e.g. *"only tested on a few datasets or with a few runs"*). Source: [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist).
- **Intervals or it didn't happen.** Same checklist: results supporting main claims should carry *"error bars, confidence intervals, or statistical significance tests,"* with the variability source, method, and SD-vs-SE choice stated. Source: [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist) (Experiment Statistical Significance item). PerfPilot's Welch-interval classification is exactly this — name the method on screen ("Welch interval, +2% uplift bar") instead of saying "verified."
- **Real regressions look like PerfPilot's problem.** Upstream vLLM issue #39940 documents a spec-decode latency regression (TPOT +8.4%, benefit halved 16.0%→8.4%) isolated by an A/B with acceptance rate unchanged (~31%) — evidence that per-commit perf archaeology is a real, unsolved maintainer pain, and that careful isolation (not vibes) is how the community already works. Source: [vLLM issue #39940](https://github.com/vllm-project/vllm/issues/39940) (fetched; PLAN.md §6 lists #39940/#48396/#61989 as motivation). Use one such number in the Problem beat; don't claim PerfPilot fixed that issue.
- **Judges reward openness.** *"The more open you can be with how you created your project, the better"* and ambiguity/detail-free code hurt scores (see §1.3). Source: [Advice from 5 seasoned judges — Devpost](https://info.devpost.com/blog/hackathon-judging-tips). A named limitation with a test beats a hidden one discovered by a judge.

### 3.2 Framing rules (apply verbatim)

1. **Lead with the invariant, not the outcome.** Headline = *"0 wrong verdicts; every claim carries an interval; cost stated per run."* Outcomes (culprit found / no regression / no uplift) are instances. This is true on both the clean local protocol and the messy GPU runs, so no table needs hiding.
2. **Name the three result types distinctly.** (a) *Bug in our harness, fixed + regression-tested* (baseline outlier). (b) *Premise failed, verdict still correct* (mild overlay −0.8%). (c) *True negative* (tune: already optimal). Conflating them reads as excuse-making; separating them reads as method.
3. **Every negative gets: expectation → measurement → decision.** "We expected ~10% (PLAN keep rule); we measured −0.8% [interval]; therefore `no_regression` is correct and the overlay gets replaced." Same shape for tune: "bar was interval > +2%; best candidate didn't clear it; baseline kept, quality held."
4. **Show the guardrail working.** The fp8-on-A100 rejection and the first-commit-undetectable limitation (README: driver *"honestly reports `no_regression` instead of fabricating a culprit,"* locked by test) are proof the system refuses to hallucinate wins. Say that sentence out loud.
5. **Never merge the two eval tables into one score.** Present local (protocol correctness, fake transport, $0) and GPU (real A100, $4.63) as different instruments answering different questions. Merging them is the one move that converts rigor into a credibility loss if a judge cross-checks `runs/`.
6. **Cost is part of rigor.** State GPU minutes and dollars next to each GPU claim (README already does: eval $4.63, tune ~$1.60). NeurIPS-style reproducibility + Devpost "more details" both reward it; it also pre-empts "can judges rerun this?" — answer: reports re-render offline, video is the recorded proof (PLAN.md risk 5).

**One-sentence honest headline (safe to use everywhere):**

> "On real A100 runs PerfPilot made zero wrong calls: it caught its own baseline-outlier bug and fixed it with a regression test, correctly reported no-regression when our mild-overlay premise turned out to measure −0.8%, and kept the baseline when no serve flag cleared a +2% interval bar — because every claim has to survive its interval."

---

## 4. X-post and README narrative patterns

### 4.1 README / submission text (Devpost "Project Story" pattern)

Devpost's judging guidance says video and text must *complement* each other — a great video with a lacking description (or vice-versa) loses points (see §1.3). Source: [Advice from 5 seasoned judges](https://info.devpost.com/blog/hackathon-judging-tips). Devpost submission forms conventionally ask for a Project Story covering inspiration, what it does, how you built it, challenges, learnings, and what's next — the BeachHacks/Devpost template enumerates: *"Inspiration? What does it do? How did you build it? What challenges did you come across? What are you proud of? What did you learn? What comes next?"* (pattern visible across Devpost hackathon pages; the judging-criteria summary is at [Understanding hackathon submission and judging criteria](https://info.devpost.com/blog/understanding-hackathon-submission-and-judging-criteria)). Winners' announcements follow the same spine in miniature (§1.4).

Recommended README/submission order (PerfPilot's current README already has most blocks; reorder/add the starred items):

1. **Title + one-line impact** (present). Add the Push-the-Frontier sentence.* (“Push the Frontier: an agent that gives vLLM maintainers a new capability — per-commit regression culprits with statistical proof.”)
2. **One-sentence usage** for non-expert judges* (`perfpilot bisect --good G --bad B` → culprit + interval + GPU $; `perfpilot report runs/<id>` re-renders offline).
3. **Problem with external proof** (vLLM #39940-style regression, §3.1) — 3 sentences max, before architecture.*
4. **How it works** (present: two planes, brain advises / stats disposes). Keep the diagram link.
5. **Eval — local protocol table, then GPU table, then tune paragraph** (present). Add the §3.2 framing sentences so a skimming judge doesn't read the GPU table as failure.*
6. **Limitations** (present: first-commit case). NeurIPS-style separate section — keep and title it exactly "Limitations." Source: [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist).
7. **Cost + reproducibility** (present). State total billed, per-run artifacts, offline re-render.
8. **Hackathon eligibility** (present: open weights, agentic loop with accept/reject log). Repeat verbatim in submission text — judges filter on requirements (§1.3).
9. **Built during** (present).

Style rules from the sources: err on more detail in text (video stays tight); every claim links to `runs/<id>/report.html` or a test name; no number without interval/denominator/cost; challenges section = the three §3.2 result types, proudly.

### 4.2 X post

No official X-post rubric exists in the fetched primary sources — treat X as a *distribution* surface for the bonus/community signal (the rules mirror notes bonus prizes *"such as a public vote or the most viral project"* — source: [madrobot.blog rules summary](https://madrobot.blog/2026/10/03/meta-global-ai-developer-hackathon-1-million-prizes-muse-model-api/)). PLAN.md §5 already specifies the format: *impact line + 20s dashboard clip + repo link.* Patterns that follow from the video/README evidence:

- **Post 1 (launch, pin it):** impact line (the §3.2 honest headline, shortened) + 20s clip of the bisect narrowing to culprit + repo link + "#MetaAIHackathon / Push the Frontier" + models named (Llama-3.2-1B, muse-spark-open). One idea only; thread carries detail.
- **Post 2 (rigor thread, 3–4 posts):** (i) baseline-outlier screenshot `[3569, 3594, 1703]` → fix + test name; (ii) mild-overlay −0.8% premise-failure; (iii) tune negative (5 no-uplift, fp8 rejected on A100, baseline kept). Each ends with the artifact link. This thread *is* the "open about how you created it" signal judges reward (§1.3).
- **Post 3 (numbers card):** eval 2/2 local, 0 wrong GPU, $4.63 + $1.60, 162 tests (per README) — as a single image/table, not a claim without denominators.
- **Do:** post the clip natively (not a YouTube link), alt-text the numbers, reply to vLLM/maintainer accounts with the #39940 context (don't tag-spam judges). **Don't:** promise an uplift number, post a cherry-picked single trial, or thread before the repo README matches the posts.

Timing: X post at submission (checklist item), rigor thread within the judging window (Nov 4–8) when judges may search the project name. Sources for window: [Cerebral Valley event page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon).

---

## 5. Recommended PerfPilot outline (apply directly)

### 5.1 Video script outline (2:50 target, recorded reports only, $0 new GPU)

Replaces PLAN.md §5 / docs/video-shots.md beats where they promise an uplift. Word budget ≈ 400 words.

| Time | Beat | On screen (recorded) | Voiceover (draft, adapt) | Rubric hit |
|---|---|---|---|---|
| 0:00–0:15 | Hook | `runs/m2-eval-9/report.html` verdict card; tok/s cliff | "Last deploy cost a third of our throughput — and no one could say which commit did it. This is PerfPilot." *(Use 32% only if that run's interval supports it; else "cost us a cliff in tok/s.")* | Impact |
| 0:15–0:40 | Problem | vLLM issue #39940 numbers (16.0%→8.4% benefit halved); commit-range diagram (`docs/architecture.html`) | "vLLM regressions like this ship silently. Maintainers bisect by hand across dozens of commits, one GPU benchmark at a time — slow, expensive, and easy to fool with one noisy run. We know; our first GPU eval was fooled exactly that way." | Impact, problem-behind-solution |
| 0:40–0:55 | Solution | Two-plane diagram; one-line commands | "PerfPilot is an agent for Push the Frontier. PerfBisect finds the first slow commit; TuneSmith tries to win speed back with serve flags. The brain proposes — benchmarks dispose. Every claim carries a statistical interval, or it doesn't ship." + "Every model in this loop is open weights." | Creative Meta-AI use, theme |
| 0:55–1:35 | Proof: bisect | Report timeline narrowing; culprit + diff; Spark explanation panel; `events.jsonl` accept/reject scroll | Walk one narrowing step: probe → Welch interval → narrow. Point at the proposal log: "accepted, rejected — the agent doesn't get the final word, the interval does." Show the `[3569, 3594, 1703]` baseline-outlier moment and the re-measure fix in one breath. | Demo (largest block), technical depth |
| 1:35–2:00 | Proof: tune (honest) | Tune report: 6 candidates, keep/revert, fp8 rejection | "TuneSmith tried six flag sets on an A100. Five didn't clear our +2% interval bar; fp8 can't run on A100 and was recorded as a rejection. So it kept the baseline and held quality. An agent that can say 'no win here' is one you can trust when it says 'win.'" | Demo, technical depth |
| 2:00–2:35 | Numbers | Eval tables (local + GPU side by side, labelled); cost line | "Local protocol: 2 of 2 culprits, zero wrong. Real A100 eval, $4.63: zero wrong verdicts — including correctly reporting no-regression when our mild-overlay premise measured minus 0.8%, no effect to find. Tune run, about $1.60: honest negative. Limitations on screen and in the README — a culprit in the very first commit is undetectable without a pre-range baseline, and we say so." | Technical depth, impact |
| 2:35–2:55 | Close | End card: repo, `perfpilot report runs/<id>`, X handle | "PerfPilot: per-commit culprits with proof, flag tuning that only keeps verified wins, and a paper trail — `report.html` plus `events.jsonl` — that re-renders offline, no GPU needed to check our work. Built this week, all open weights. Thank you." | Demo, eligibility |

Production notes: record `report.html`/`architecture.html` at 125% zoom; captions burned in (numbers must be readable paused); no live GPU (§2.1); upload unlisted→public and test the link logged-out before submitting; reserve the final afternoon solely for this (winner advice, §1.4).

### 5.2 Submission text outline (Devpost/Cerebral Valley form + README top)

1. **Title:** PerfPilot — the agent that finds the commit that slowed your vLLM deployment (and proves it).
2. **Elevator pitch (2 sentences):** Push-the-Frontier problem + agent/interval mechanism + open-weights sentence.
3. **Problem:** §3.1 external proof + hand-bisect cost.
4. **What we built:** PerfBisect / TuneSmith, two planes, brain-advises/stats-disposes, fallback with no key.
5. **How we know it works:** local table → GPU table → tune paragraph, using §3.2 framing verbatim; link each row to `runs/<id>/`.
6. **Challenges (wear them proudly):** three result types from §3.2, each expectation→measurement→decision.
7. **Limitations:** first-commit case + mild-overlay replacement note.
8. **Cost & reproducibility:** totals, per-run artifacts, offline re-render command.
9. **Eligibility:** open weights (Llama-3.2-1B measured, muse-spark-open brain), agentic loop (proposal log in `events.jsonl`), built-during statement.
10. **What's next:** replace mild overlay, pre-range baseline for position-0 culprits, wider flag space on Hopper.

### 5.3 Checklist deltas vs PLAN.md §5 (do these, in order)

- [ ] Rewrite the 0:55 beat: delete "verified uplift number," insert §5.1 honest-tune beat.
- [ ] Decide the hook number from the actual run interval; if unsupported, use the qualitative cliff line.
- [ ] Add Push-the-Frontier + one-sentence-usage lines to README top (§4.1 items 1–2).
- [ ] Add §3.2 framing sentences above the GPU eval table; keep both tables, never merge.
- [ ] Script → record → caption → upload → logged-out link test, in one reserved block; then X Post 1.
- [ ] Verify every video sentence matches README/`runs/` (judge ambiguity check, §1.3).

---

## Sources (all fetched and read)

1. [Meta Global AI Developer Hackathon — Cerebral Valley (official event + rules)](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)
2. [Meta Global AI Developer Hackathon — dev.meta.ai](https://dev.meta.ai/events/global-hackathon)
3. [Meta is offering $1 million in prizes… rules summary — madrobot.blog](https://madrobot.blog/2026/10/03/meta-global-ai-developer-hackathon-1-million-prizes-muse-model-api/)
4. [Understanding hackathon submission and judging criteria — Devpost](https://info.devpost.com/blog/understanding-hackathon-submission-and-judging-criteria)
5. [6 Tips for making a winning hackathon demo video — Devpost](https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video)
6. [How to win a hackathon: Advice from 5 seasoned judges — Devpost](https://info.devpost.com/blog/hackathon-judging-tips)
7. [Meet the winners of the Built with Opus 4.7 hackathon — claude.com](https://claude.com/resources/articles/meet-the-winners-of-built-with-opus-4-7-claude-code-hackathon)
8. [Meet the winners of our Built with Opus 4.6 hackathon — claude.com](https://claude.com/blog/meet-the-winners-of-our-built-with-opus-4-6-claude-code-hackathon)
9. [Meet the winners of our first-ever LlamaCon Hackathon — ai.meta.com](https://ai.meta.com/blog/llamacon-hackathon/)
10. [NeurIPS Paper Checklist — neurips.cc](https://neurips.cc/public/guides/PaperChecklist)
11. [vLLM issue #39940 — Eagle3 latency regression — github.com](https://github.com/vllm-project/vllm/issues/39940)

*Method note: search results were used only to discover sources; every claim above traces to a page fetched with the fetch tool and read beyond the snippet. Where the Cerebral Valley page rendering truncated the weighted rubric, the weights are cited to the rules mirror (source 3) and flagged as such in §1.1.*
