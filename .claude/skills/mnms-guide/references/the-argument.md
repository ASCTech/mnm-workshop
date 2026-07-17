# The argument: what we saw agents unlock

This is the case the workshop makes, in short form. It isn't a theory we're
defending so much as a pattern we kept noticing — so treat it as *here's what we've
seen*, and feel free to push back on it with your own experience.

## Where this comes from

Our team spends its time helping researchers turn a *method* into working software:
taking a corpus — PDFs, audio, syllabi, survey transcripts — and getting structured
data back out that they can analyze. We've done this across a lot of fields, and no
two projects wanted the same thing. That variety is actually the point. The specific
tool almost never transfers; what transfers is the *process* of getting from "I have
this corpus and this question" to "I have data I can trust." The branches in this
repo are a few of those projects, rebuilt on public data so we can share them.

The common signature, when it appeared, looked like this: Python + `uv`, LLM calls
through one LiteLLM proxy, a few models compared, cost tracked, and a report at the
end. **Corpus in → structured data out.** But that's a description of what worked,
not a template we hand people.

## Two baselines worth keeping separate

When someone asks "couldn't the researcher just do this by hand?", it helps to split
two very different things:

1. **Doing the task by hand** — coding 1,000 PDFs, transcribing hundreds of clips.
2. **Building the tool to do it, without a software team.**

In our experience, agents mostly collapsed the *second* cost, not the first. That's
the part we find worth talking about, because it's the part that used to quietly kill
the ambitious version of a project.

## Three kinds of change we noticed

These aren't tiers in a ranking so much as three distinct things that happened,
roughly in increasing order of how much they surprised us. Each branch happens to
illustrate one cleanly:

1. **Some work was just expensive, and got cheap.** Grunt work that was always
   possible, only costly — months of RA time compressed into a days-long round trip
   for a few dollars.
   → **`corpus_coding`** is the clearest version of this: the same codebook run
   across many models and repeated runs, for the price of a coffee. (See the cost
   spread in `reading-results.md`.)

2. **Some work had been out of reach for the researcher alone** — not because of
   effort, but because it needed a computational-methods skill they didn't have and
   couldn't easily hire for.
   → **`structure_analysis`** is this one: corpus-wide topic clustering
   (extract → embed → BERTopic) isn't something most humanities or social-science
   researchers assemble on their own.

3. **Sometimes the bottleneck was the code, not the compute — and some designs were
   simply infeasible by hand at all.**
   → **`bt_scoring`** is the case that most changed how we think about this:
   thousands of pairwise judgments no human could hold in mind. It only works because
   the judge is *cheap and consistent*. The unlock there is reliability, not raw
   capability.

## The through-line: package it so you can invoke it, not rebuild it

The most durable win we've seen isn't a clever prompt or a specific model. It's
taking a hard-won capability and wrapping it in a stable chassis with a small, safe
surface — so the next study, or the next researcher, invokes it instead of rebuilding
it from scratch. That's what turned one-off projects into things we could reuse.

A Claude Code skill is exactly that kind of packaging, for a coding agent. The
`mnms-guide` skill you're reading is a small worked example of it — and
`authoring-skills.md` closes the loop by walking through how to package *your own*
method the same way.
