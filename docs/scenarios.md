# Scenarios: which agents this helps, and which it does not

The headline number for this router is -22.5% of everything one A/B run was billed for. That was
measured on one agent, with tools this project wrote and conversations this project authored, and
the honest reading of it is "this worked here". It is not an answer to your question.

The second number is the one to read next to it. On a hundred tools nobody here wrote, driven by
queries nobody here wrote, the saving was -13.4%, and it came with a measurable drop in how often
the agent called the tool the question needed. There is a second mode that gives the answer quality
back and saves 2.0% instead. That is a choice you get to make, and both sides of it are measured in
[Very large surface](#very-large-surface).

Your question is which of these shapes your agent is. This page is a book of shapes. Each one says
what the shape looks like, what the router did on it, and whether you should switch it on. Some of
the shapes are losses, and those are the useful ones. A page that only listed wins would be
marketing.

Every number here is measured unless the section says otherwise, and the sections that are not
measured say what would settle them.

## Finding your shape

Start with `smart-router qualify`. It reads your tool schemas and one representative prompt, and it
answers before any of GCP is provisioned. Then read the section it points you at.

| Shape | You have | Verdict |
| --- | --- | --- |
| [Small surface](#small-surface) | fewer than 20 tools | The router declines. Do not install it. |
| [The trap](#the-trap) | a prompt that caches now and would stop caching if it shrank | Measured loss of 25.5%. `qualify` refuses. |
| [Broad support agent](#broad-support-agent) | around 100 tools, multi-turn sessions | Measured win of 22.5%. This is the headline. |
| [One-shot sessions](#one-shot-sessions) | one model call per session | Works, with a separate safety rule. |
| [Very large surface](#very-large-surface) | 100 tools or more, none dominant | Measured win of 13.4% that costs answer quality, or 2.0% that does not. |
| [Document tools](#document-tools) | tools that return kilobytes | Barely matters. 14x heavier payloads cost half a point. |
| [Drifting traffic](#drifting-traffic) | a tool surface that changes under you | Costly. Recall fell 0.98 to 0.62 when the surface quadrupled. |
| [Sparse and cold](#sparse-and-cold) | too little traffic to keep a prefix warm | Guarded. The guard was right on five of five real losses. |
| [Sub-agents and MCP](#sub-agents-and-mcp) | tools that are not local functions | Works. The saving has not been billed on one. |

## Small surface

**The shape.** A handful of tools. A classifier with two, a retrieval agent with five, a workflow
agent with a dozen.

**What happens.** The router declines at install. `min_tools_to_route` is 20, and below it the
router costs more to run than the declarations it could remove are worth. There is no configuration
that makes a two-tool agent worth routing, because there is nothing to prune.

**Should you switch it on.** No, and `qualify` will tell you so without you having to provision
anything.

## The trap

**The shape.** Your prompt is above the model's [context-cache
floor](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/context-cache/context-cache-overview)
today, and pruning the tool declarations would push it below. Gemini Flash's floor is 2,048 tokens;
Gemini 3's is 4,096.

**What happens.** You lose money, and you lose a lot of it. Removing tokens turns caching off
entirely, and the tokens you kept go from billing at one tenth to billing at full rate. Measured at
**+25.5%**, which is the largest loss in this project's records. A prompt that drops from 2,100 to
2,000 tokens is 4.8% smaller and 7.7 times more expensive.

**Should you switch it on.** No. This is the one band `qualify` refuses outright rather than warns
about. If you are near the floor, the fix is to grow the prompt above it, not to prune below it.

## Broad support agent

**The shape.** Around 100 tools, sessions that run four or five turns, a first message that states
what the user wants. This is a customer support agent, an internal helpdesk, an operations
assistant.

**What happens.** The router predicts the tool set from the opening message, prunes the declaration
list once for the whole session, and the session runs normally on a smaller prompt.

Measured over 574 sessions split evenly between routed and holdout arms:

| | holdout | routed | change |
| --- | --- | --- | --- |
| billed input per turn | 2,323.2 | 1,780.3 | -23.4% |
| billed input per session | 10,771.8 | 8,314.3 | -22.8% |
| output per session | 156.3 | 154.6 | -1.1% |
| **whole system per session** | **10,928.2** | **8,468.9** | **-22.5%** |
| turns per session | 4.637 | 4.670 | +0.7% |

The turn count is the one to look at hardest, because it is the number that would move if pruning
were hiding tools the session needed. It did not move: +0.033 turns, Welch's t of 0.33, two-sided
p = 0.745. Sessions did not run longer, so the per-turn saving is a real saving and not a division
artefact.

**Should you switch it on.** Yes, and ramp it. A 200-session A/B split evenly between arms is
enough to confirm the effect on your own traffic at this effect size, which is about twenty minutes
of running.

## One-shot sessions

**The shape.** Every session is a single model call. Classify and return, extract and return, route
and hand off.

**What happens.** Pruning works the same way, but the safety net that watches for pruning going
wrong does not, and that needed a separate rule. The cache watch normally compares later calls in a
session, because a session's first call has no prefix of its own and its price depends on what
other sessions left warm rather than on anything this session did. An agent with no later calls has
nothing to compare, so no verdict ever arrives and a loss could run indefinitely.

The fallback compares first calls to first calls, with two brakes: it needs 500 samples per arm
instead of 50, and a 15% margin instead of 5%. Those numbers make the rule self-selecting. Reaching
500 first calls in both arms before either later-call arm reaches 50 requires more than nine
sessions in ten to be a single call, so a multi-turn agent reaches its ordinary verdict first and
never sees the fallback.

**Should you switch it on.** Yes, if the surface qualifies. Read `CacheWatch.verdict_basis()` in the
logs if a demotion fires: "first" and "later" mean different things, and "first" can mean your
deployment simply has not warmed up yet.

## Very large surface

**The shape.** A hundred tools or more, on a surface where no small group of tools dominates.

**What happens.** We ran this live, on somebody else's tools and somebody else's queries. The
surface is the 100 commonest tools in
[`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k),
a public function-calling benchmark of 60,000 request-and-answer pairs over real, executable
third-party APIs, built with the [APIGen](https://arxiv.org/abs/2406.18518) pipeline: the APIs are
real, and the requests are model-written and then verified by actually calling them. The traffic
here is that corpus's own queries. Nothing in the fixture was written here except a system
instruction that names no tool.

Over 599 sessions split into arms, the whole-system bill fell **13.4%**. That is a real saving and
it is smaller than the 22.5% on our own agent, for two reasons worth knowing. xLAM's schemas are
terser, so there is less prompt to give up. And the routed arm's cache hit rate was 66% against the
holdout arm's 91%, because 24 menus over 295 routed sessions is twelve sessions a menu and the
prefixes never warmed up. That second one is the [sparse and cold](#sparse-and-cold) shape showing
through, not a property of wide surfaces.

The corpus also ships an answer key, which no other run in this book had, so this is the one place
we can say what the saving cost:

| | holdout | routed |
| --- | --- | --- |
| called every tool the query needed | 87.2% | 75.6% |
| called nothing at all | 6.8% | 10.5% |

That gap is real, not noise: p = 0.003. **The saving was not free.** It was bought by shipping an
artifact whose menus held 91% of what sessions ask for, and the missing 9% turned into sessions
where the model had nothing to call and declined.

The useful part is that this was predictable and was predicted. `publish_router.py` refuses an
artifact whose menu recall is below a bar, and it measured this one at 0.9095 before anything ran.
One minus that is 0.0905; the measured loss in answer recall was 0.0810. **Menu recall at publish
time is a direct estimate of the share of requests the router will fail.** So the bar is not a
statistic, it is a dial: it sets how many requests you are willing to fail to save tokens. The
shipped default of 0.90 is one in ten.

There is a ceiling on how far you can turn that dial, and it is the reason the default has not
moved. On this surface menu recall stops improving at 0.9265 and stays there whether the catalogue
holds 40 menus or 96. The model itself scores 0.9960 here when it answers per query, so the eight
points between are the cost of serving from a fixed catalogue at all. Raising the bar past 0.93 on a
wide surface does not produce a better artifact, it produces no artifact.

**So the real dial is the catalogue, and we priced turning it off.** `--no-menus` drops the pooled
catalogue and prunes each session on its own prediction. Same 600 queries, same booster, one day
later:

| | pooled menus | one prune per session |
| --- | --- | --- |
| whole-system saving | **13.4%** | 2.0% |
| called every tool the query needed | 75.6%, against 87.2% holdout | 88.6%, against 90.3% holdout |
| is that gap real | yes, p = 0.003 | no, p = 0.714 |
| tools left in the prompt | 11 | 6.5 |

Per-session pruning prunes harder and costs nothing you can measure in answer quality. It also gives
up eleven of the thirteen points of saving, and the reason is the cache. Its prompt is 61% smaller
than the holdout arm's and its bill is only 6% smaller, because a cached token bills at a tenth and
every session now has a prefix no other session shares. A prune that breaks caching has to remove
roughly ten times what it un-caches before it wins anything. Declarations were two-thirds of the
prompt here, so even deleting all hundred of them does not get there.

That is the whole trade on a wide surface: about eleven points of your bill against about eight
points of "the user got what they asked for". Nothing in the middle has been measured, and we do not
believe there is much middle to find, because the catalogue either exists and caches or does not.

Beyond 100 tools we have a scoring pass rather than a live run. `xlam-function-calling-60k` holds
60,000 queries over 3,179 tools:

| surface | tools | queries | keeping the 11 most popular | the router | tools kept |
| --- | --- | --- | --- | --- | --- |
| top 25 | 25 | 4,587 | 0.5690 | 0.9991 | 1.3 |
| top 100 | 100 | 15,786 | 0.1843 | 0.9960 | 2.0 |
| top 400 | 400 | 28,676 | 0.1056 | 0.9943 | 2.7 |
| every tool | 3,179 | 60,000 | 0.0544 | 0.9666 | 5.7 |

Two things to take from this. The prediction holds as the surface grows, reaching 96.7% recall at
3,179 tools while keeping under six of them. And a static menu of the most popular tools is
worthless at that scale, at 5.4%, so there is no cheap alternative to prediction on a wide surface.

What does not transfer is the menu pool. The router serves from a fixed catalogue of pre-computed
menus so that prompt prefixes stay cacheable, and pooled recall falls from 0.9908 at 25 tools to
0.2298 at 3,179. The live run above is what that looks like in practice at 100 tools, and it is why
the quality table has the numbers it has. A very large surface needs more menus than the catalogue
currently holds, and sizing that is open work.

Menus are built one of two ways, and which one you pick matters most on exactly this shape. The
default, `merged`, reaches menu recall 0.9095 on the 100-tool surface. `padded` reaches 0.6942 at
the same size and count, which the publish gate refuses. So you should not meet that refusal; if
you do, `--menu-candidates padded` is the other arm, and it suits a narrow surface better.

**Should you switch it on.** Yes on the prediction, and then pick a mode deliberately rather than
taking the default because it is the default.

Ship the pooled catalogue if a tenth of requests going unserved is a cost your product can carry,
and read the miss rate `publish_router.py` prints at publish time as exactly that. Ship `--no-menus`
if it cannot. On a wide surface you will publish near the bar rather than comfortably above it, so
this is a decision you will actually be making, unlike a narrow-surface agent that clears 0.97
without trying.

If you have zero training rows on that wide surface, `judges=("prototype",)` reaches **0.9894 set
recall** (`97.8%` complete sessions, **0.8831 token efficiency**) across the 2,000 logged `xlam`
sessions at `k=15` from the tool docstrings and JSON schemas alone. Once trained, pairing the two
(`judges=("xgboost", "prototype")`) reaches **0.9927 recall** (`98.7%` complete) on the 24-menu
pool and **0.9971 recall** (`99.5%` complete, **0.9442 token efficiency**) on `--no-menus`.

Either way, run the A/B on your own traffic before you trust a number on this page, and measure
answer quality alongside the tokens rather than after them. Ours is one corpus.

## Document tools

**The shape.** Tools that return real payloads. A document fetcher, a search API, a database query
that comes back with rows.

**What happens.** Almost exactly what happens without them. Tool results come back into the next
prompt, so the tools a session calls are part of what it is billed for, and there was a real
possibility that pruning would shift the payload bill enough to matter. It does not.

We ran the same agent twice, changing only what the tools returned. The heavy version's tool calls
came back with a median of 1,382 characters against the light version's 95, a p95 of 6,380 against
333, and a maximum of 9,559 against 703.

| | light payloads | document payloads |
| --- | --- | --- |
| billed input per turn | -23.4% | -21.1% |
| whole system per session | -22.5% | -22.0% |
| the tool-response channel | -0.01%, a counterfactual | +0.11%, measured |

Fourteen times heavier payloads cost half a point of saving. The response channel itself is worth
about a tenth of a percent of the bill, in the direction of costing rather than saving, which is a
rounding error against 22%.

One number in that run looked like a finding and was not. The routed arm carried 9.9% more response
characters per session while making 4.8% fewer tool calls, which reads as pruning steering traffic
toward the tools that return more. Tested, it comes out at p = 0.39. Response size per session has a
much longer tail than turn count, and a gap that size is what the tail produces over a few hundred
sessions.

There is a second question the fixtures still cannot answer. Our tools are deterministic stubs that
return in 2 ms against a 2,949 ms model call, so there is no latency to measure.

**Should you switch it on.** Yes, and the size of what your tools return is not a reason to
hesitate. If your payloads are much larger again, or if their size depends on which tool was called,
measure your own tool-response bill rather than taking the half-point from this page.

## Drifting traffic

**The shape.** The tool surface moves under the model. You add tools, retire tools, or your users
change what they ask for.

**What happens.** It depends entirely on which of those three you mean, and the difference is
large. Adding tools is the one that hurts. When this project's own agent went from 25 tools to 100
overnight, recall on the unchanged model fell from 0.98 to 0.62 on the first day of the new surface.
Users changing what they ask for is much cheaper: withholding a whole topic from training and then
serving it back cost 1.3 points of recall. Traffic simply moving to a new day, with the same tools
on offer, cost nothing measurable at all.

**Bridging the gap between retrains.** Setting `judges=("xgboost", "prototype")` scores newly
deployed and classless tools semantically against the live tool schemas instead of either carrying
every unseen tool on every menu (`0.5342` token efficiency) or dropping unseen tools blindly
(`0.6617` recall when 15 active tools are held out from `xgboost`). With `PrototypeJudge` paired
alongside `xgboost`, recall on that 15-tool holdout recovers to **0.8978** (`88.2%` complete
sessions) at **0.9496 token efficiency** on `support`, and **0.9693** (`94.4%` complete sessions) on
Salesforce `xlam`.

**How you can tell.** The share of the declared tools your model has no class for. A healthy model
on this corpus does not recognise about a quarter of its surface, because rare tools never reach the
training floor, and that share went to 77% the day the surface doubled. The two signals you might
reach for first, how confident the model is and how unfamiliar the openings look, both move sharply
under a topic shift that barely costs anything. They are worth a retrain and they are the wrong
input to an automatic shutoff.

**Should you switch it on.** Yes. Enable `judges=("xgboost", "prototype")` if you deploy new tools
between retrains, retrain when your tool list changes, and otherwise on a schedule. A retrain costs
about eleven cents, so there is no reason to stretch the interval. Record the unseen share your
model publishes with, because the number that matters is the change from it and not the number
itself.

## Sparse and cold

**The shape.** Not enough traffic per menu to keep a prompt prefix warm in the model's cache.

**What happens.** You can lose money, and the router has a guard for it that is now calibrated. A
cold prefix bills at full rate, so a smaller prompt that misses the cache can cost more than a
larger one that hits it. The cache watch compares what your routed and unrouted calls actually bill
and turns pruning off for the process when pruning is losing. Replayed over twelve logged runs and
21,000 priced calls, it fired on five of them, was right on all five, and never fired on a run where
pruning was winning. On the worst of them, a 17.6% loss, it caught the problem five minutes in.

**Where it is blind.** It needs 50 routed calls past the opening turn of a session before it will
say anything, and it counts them in memory, so a restart starts the count again. One run in this
project's records lost 13.8% and was never judged, because it only ever produced 15 such calls. If
your agent routes lightly, expect the guard to abstain rather than protect you.

**Should you switch it on.** Cautiously. Run the A/B, and give it enough sessions that the prefixes
have warmed up. One run in this project's records saw turn-one cache share climb from 0.31 to 0.66
across a single run, so an early reading can look like a loss that is really a cold start.

## Sub-agents and MCP

**The shape.** Tools that are not local Python functions. An MCP server, a sub-agent wrapped in
`AgentTool`, a sub-agent reached by transfer.

**What happens.** They are pruned like anything else, because by the time the router sees a request
ADK has already flattened every toolset into one list of function declarations with no record of
where any of them came from. A tool that was pruned still works if the model calls it anyway, over
MCP as well as locally. `never_prune` accepts an MCP name. A parent agent's decision cannot reach a
sub-agent's tool list, under either delegation style, because the routing key carries the agent
name. `transfer_to_agent` is never pruned at all: it is the control plane the tree moves on, and no
model is ever trained to name it.

**Where it is blind.** One hazard is worth knowing about. If you set `tool_name_prefix` on an
`McpToolset`, your live tool names stop matching the names the model was trained on, and nothing
warns you. The router passes unclassed tools through rather than dropping them, so your agent keeps
working, but every renamed tool now rides along in every session and the saving shrinks. Republish
after the rename and it goes away. Set `prunable_by_default` if you need those tools gone before
then. The same thing happens whenever a new server appears between retrains, and it is the safe
direction to fail in.

The other gap is money. All of this is correctness, tested against a real MCP server over a real
subprocess. No A/B has been run on an agent whose tools mostly come over MCP, so the saving on that
kind of surface is inferred from declaration counts rather than measured on a bill.

**Should you switch it on.** Yes, on the same terms as any other surface: run `qualify`, which reads
MCP schemas as well as local ones, and then A/B it. Nothing about the origin of a tool changes the
arithmetic.
