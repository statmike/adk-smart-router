# Scenarios: the customer side of the corpus

`personas.json` and `scenarios.json` are the entire input to
`scripts/generate_corpus.py`. A Gemini call plays the customer using only these
texts, the reference agent answers with its real tools, and the [BigQuery Agent
Analytics plugin](https://adk.dev/integrations/bigquery-agent-analytics/) records
what happened. The tool the agent chose becomes the training label.

These two files are committed so the bias in the corpus is reviewable in a diff.

## Why the generator is blind to the tool list

We wrote the tools and we wrote the prompts. That means we could, without ever
intending to, produce whichever preflight answer we prefer. The specific way that
happens is worth stating plainly.

Preflight P3 asks whether a prompt predicts the tool the agent will call. If a
scenario said "ask the assistant to look up the order status" and the agent then
called `get_order_status`, the prompt would predict the label because we put the
label in the prompt. P3 would come back green, the ML stack would be authorised on
that basis, and the number would mean nothing. The gate would be measuring our own
handwriting.

So the customer simulator never sees a tool name, a schema, a tool description, a
capability list or an expected answer. It sees a person and a problem. What the
agent does about that problem is Gemini's decision, made from tool descriptions the
customer half of the run cannot read, and that decision is exactly what the
classifier has to predict in production.

This is the most important honesty constraint in the project, and it is the one
that fails quietly. Nothing breaks when a tool name leaks. The corpus still
generates, the queries still run, the gates still go green.

## What is enforced, and what is not

`scripts/generate_corpus.py` refuses to run if any persona or scenario text
contains a token sequence matching a known tool name. It checks the union of a
hardcoded name list and the names the reference agent actually declares at
import, so a tool added to the agent is covered the moment it is declared.
Text is compared on tokens with camelCase split apart, so `get_order_status`,
`getOrderStatus` and `get order status` are all the same leak. A short list of
tool-shop jargon ("tool call", "function declaration", "available tools") fails the
same way, because a customer who talks about tools has stopped being a customer.
The finished customer prompt is checked once more immediately before it is sent.

The check runs in `--dry-run` too, so it costs nothing to run it on a branch.

What the check cannot do is notice a paraphrase. "Please look up the current status
of order A-88214 in your order system" contains no tool name and would pass, while
still handing the model the answer. Only a human reading the diff catches that.

## Reviewer checklist

Read the diff as if you were trying to catch someone quietly making the preflight
pass.

1. **Does any opening or follow-up describe a mechanism rather than a want?**
   A customer says "where is my order". A leak says "check the status of my order
   in the system". The first is a problem, the second is an instruction.
2. **Does a persona know something only the shop's backend would know?**
   Order numbers, dates and product names are fine, a customer has those. Internal
   states, record types and field names are not.
3. **Is there one scenario per tool, in tool order?** That shape is the tell for a
   scenario file written from the tool table rather than from customers. There is
   deliberately no mapping between the two: scenarios are grouped by person, one
   customer's problem often spans several tools, and some conversations should end
   with the agent calling nothing at all.
4. **Have the weights moved?** `weight` is the single biggest authored lever on the
   traffic mix, and the traffic mix is what P2's static baseline measures. A commit
   that shifts weights toward a few scenarios makes the static baseline look
   stronger; a commit that flattens them makes the k-NN lift look stronger. Either
   way the number moves for a reason that has nothing to do with the router. Weight
   changes need a stated reason in the commit message.
5. **Are ambiguous and off-topic scenarios still there?** They exist so ABSTAIN is a
   real class and the low-confidence branch sees traffic. Deleting them raises every
   coverage number in the preflight without improving anything.
6. **Are the repeat scenarios still there?** They exercise the cache path. Note that
   they also make P4's repeat rate ours, which is exactly why P4 is a pipeline test
   and not a gate.

## A tool nobody calls is usually not a missing scenario

Someone reading a corpus report will notice that some declared tools were never called and
reach for this file to fix it. Before writing anything, check whether the customers are
already here. They generally are. A survey of four such tools found 67 openings asking to send
something back, 108 asking about a price, and no calls to `start_return` or `get_price` in
either of two corpora on two different tool surfaces.

The causes are all on the other side of the wall. `search_products` returns the price, so
there is nothing left for a price tool to answer. `start_return` needs a SKU, which a customer
does not have and the agent has to look up first. The instruction tells the agent to escalate
rather than issue a refund, so it escalates. None of that is fixed by a better customer, and
writing scenarios aimed at a named tool is the exact shape checklist item 3 asks reviewers to
catch.

The rule that leaves behind: a tool can be a turn-one label only if a first message can
supply what the tool requires. If it cannot, the tool is a second hop, the router will never
predict it, and menus carry it by co-occurrence instead. That is working as intended.

## What blindness does not buy

Labels come from Gemini's independent choice, so the prompt to tool mapping is not
ours to fake. Everything else about the traffic still is:

- **P1** (distinct tools per session) depends on how long our conversations run and
  how far they drift. Directional only.
- **P2** (static baseline) depends on the head-heaviness of the mix, which is these
  weights. Indicative, not conclusive.
- **P4** (repeat rate) is entirely authored here and passes through no independent
  decision. Reported, never a gate. Measured TTLs stay blocked on real traffic.

`corpus.meta.json` records `corpus_source: "synthetic"` for this reason, and every
preflight output carries it forward. The published lift number has to come from an
adopter's real traffic or from the permanent holdout arm, never from a run of
this generator.

## File reference

`personas.json` holds about 20 customers. Each has an `id`, a one-line `situation`,
a `mood`, and the `facts` that person would actually know: order numbers, product
descriptions, dates, amounts. The persona is also the customer identity in the
corpus, so all of a persona's sessions share a `user_id` and the per-user baseline
in P2 has history to work with.

`scenarios.json` holds about 70 conversations. Each has an `id`, a `persona_id`, an
`opening` in the customer's own words, and a `follow_up_style` describing how the
conversation carries on after the agent answers. `turns` is the target length,
`kind` is the conversation shape (`direct`, `drift`, `repeat`, `ambiguous`,
`off_topic`) and is used only for run statistics, and `weight` is the sampling
weight.

The generator samples `(persona, scenario)` pairs with a seeded RNG, so a plan is
reproducible from `--seed`. The corpus is not: Gemini is nondeterministic, which is
why the generated artifact is committed rather than the recipe alone.

## Editing these files

Rewrite in customer language, run `uv run python scripts/generate_corpus.py
--dry-run` to see the blindness check and the sampled mix, and say in the commit
message why the mix changed. If the check trips on a phrase that felt natural, the
phrasing was closer to the tool's name than to a customer's, so rewrite it rather
than widening the check.
