# MemBridge demo video: script (about 4 minutes)

**How to read this.** Say the **SAY** lines in your own voice (small changes are fine). Between scenes, stay silent for **two full seconds**: that is how the editor finds the cut points.
Scenes 1, 2, 4, 5, 6 are **voice-over** on visuals I generate. Scene 3 is **your live screen recording** of ChatGPT and the terminal.
Teleprompter version: open `teleprompter.html` in a browser. Step-by-step recording: `RECORDING_GUIDE.md`.

---

## SCENE 1: The problem (about 20 s) · voice-over
*On screen: title card, then the three problems.*

**SAY:**
Every AI assistant now keeps a memory about you. But it is a list you can't read, can't correct, and can't take with you. And it will happily answer questions you never gave it the answer to.
I built MemBridge so that memory is yours: on your own machine, with proof of where every fact came from, and an honest "I don't know."

*(2 seconds of silence)*

## SCENE 2: A new user's first minute (about 35 s) · voice-over
*On screen: a terminal installs and starts it; then the dashboard's "Ask your memory" box.*

**SAY:**
This is what a new user sees. One command to install, one to start. No cloning, no Node, no Docker.
It loads a made-up person and answers questions. Ask where this person works: it answers, and shows that one line came from ChatGPT's summary, so it is kept as history, while what the user said themselves stays current.
Now ask for a favourite pizza topping. Nothing was ever stored, so it says so, instead of guessing.

*(2 seconds of silence)*

## SCENE 3: The real thing, with my own ChatGPT (about 70 s) · LIVE recording, your voice
*On screen: you, on chatgpt.com, then the terminal. Follow `RECORDING_GUIDE.md` section B exactly. Talk as you do it.*

**SAY (guide, in your own words):**
Now the real thing, with my own ChatGPT. ChatGPT already remembers things about me.
I run `membridge dump-prompt`, and paste it into ChatGPT. It lists everything it knows about me.
*(while it generates: "I'll speed this part up.")*
I copy that, save it, and import it. MemBridge stores each line as ChatGPT's belief, not as something I said. And the health and money lines are marked private automatically, so you won't see them here.
Let me ask something I did tell it: **<your first question>**. It answers, and shows where it came from.
Now something nobody ever told it: what is my favourite pizza topping? Nothing stored, and it doesn't guess. That "no" is the feature.

*(2 seconds of silence)*

## SCENE 4: Why a graph, and where FalkorDB fits (about 50 s) · voice-over
*On screen: the architecture diagram, the Graph tab of the dashboard, then the "no shared words" card.*

**SAY:**
Under the hood, memory is a graph in FalkorDB: people, projects, decisions, conversations, and the exact messages each fact came from.
Here is why that matters. Ask: what did I change after the ZX0 timeout? No stored fact shares a word with that question, so text search finds nothing. But the graph hops from the message, to its conversation, to the project, to the decision.
On my benchmark, text search answers none of those twelve questions. The FalkorDB graph answers all twelve.
I use FalkorDB for Cypher paths I can explain to the user, full-text and vector indexes, and PageRank and community detection for the big picture. SQLite stays the source of truth, so the graph is rebuildable, and answers are identical if it is down.

*(2 seconds of silence)*

## SCENE 5: Trust and privacy (about 40 s) · voice-over
*On screen: terminal output of the trust demo, then the dashboard unlocking private memories.*

**SAY:**
A memory has to be trustworthy. If ChatGPT's summary and I disagree, I win, and its version is kept as history.
Health, money and family facts are hidden from every assistant by default. Here an assistant tries to grant itself access: refused. I set a PIN, and give this one assistant ten minutes and two reads.
When an assistant wants something forgotten, it can only ask. I confirm, and the text is gone from the database file, not just hidden.

*(2 seconds of silence)*

## SCENE 6: Try it, and the proof (about 25 s) · voice-over
*On screen: the live sandbox, the numbers card, the install command and links.*

**SAY:**
You can try it right now. There is a read-only sandbox on made-up data, or install it with one command.
It has three hundred and fifty-seven automated tests, green on Linux and Windows, and two independent security reviews, with every finding fixed.
MemBridge: memory you own, that knows what it doesn't know.

---

### Facts used in the script (all measured; do not embellish)
- 12 of 12 vs 0 of 12: the "no shared words" benchmark questions, graph vs text-only (`docs/EVAL.md` §1).
- 357 tests; CI green on Linux and Windows (GitHub Actions).
- Two independent adversarial reviews by a separate model; every reported finding fixed with a regression test.
- Not claimed: accuracy on real data (never measured), speed records, anything about other products.
