# Getting started (about 5 minutes, no coding)

MemBridge keeps what your AI assistants know about you in one place **on your own computer**, and lets assistants read it under rules you set.
Nothing is uploaded anywhere. You need Python 3.11+ ([uv](https://docs.astral.sh/uv/) installs it for you). Docker is optional (only for the graph tab).

## 1. Try it first, on made-up data (1 minute)
```bash
uvx --from git+https://github.com/chinchoooo/membridge membridge quickstart --demo
```
It builds a small store for an invented person in `~/.membridge-demo` (your real one, if you have one, is not touched), asks it three questions, and opens the dashboard:

```
[3/6] What it knows
  6 conversations, 18 memories (4 old values kept as history, 4 private).
  Q: What database does Atlas use now?
     -> [current, user_stated] I switched Atlas from PostgreSQL to SQLite because running a server on my laptop is too heavy.
  Q: Where do I work?
     -> [current, user_stated] I work at Acme Robotics.
     -> [superseded, provider_stated] Alex Rivera works at Globex Labs.      <- ChatGPT's summary disagrees; yours wins, theirs is kept as history
  Q: What is my favourite pizza topping?
     -> (nothing stored about that, and it does not guess)
```
Prefer not to install anything? The hosted sandbox shows the same thing in a browser on made-up data (read-only): see the link in the README.

## 2. Use it on your own history
```bash
uvx --from git+https://github.com/chinchoooo/membridge membridge quickstart
```
It asks for your export, previews it (nothing is written until you say yes), imports it, shows what it learned, offers to set a PIN, and opens the dashboard.

**Getting your export.** ChatGPT: Settings > Data controls > Export data (the email can take days). Claude: Settings > Privacy > Export data.
**While you wait,** ask ChatGPT what it already remembers: run `membridge dump-prompt`, paste the text into ChatGPT, save its reply as `chatgpt-dump.txt`, then
`membridge seed-text chatgpt-dump.txt --dry-run` and again without `--dry-run`. It is imported as *ChatGPT's belief*, never as "you said it".

**Expect few facts at first.** It only keeps things you plainly said about yourself or your work, so a history of coding prompts yields little. That is deliberate: a wrong memory is worse than a missing one.

## 3. Connect an assistant (optional)
`quickstart` prints the command for Claude Code. Other assistants that speak MCP (Cursor, VS Code, ...) are in [CLIENTS.md](CLIENTS.md).
An assistant can read your memory and add to it. It cannot mark things as verified, settle disagreements, change what you stated, or forget anything without you.

## 4. Private memories
Health, money, identity-document and family facts are hidden from every assistant by default. Set a PIN once (`membridge set-pin`); to let one assistant see them for a few minutes:
`membridge grant-sensitive claude-code --minutes 10`. Mark anything it missed with `membridge mark-private <id>`. Details: [threat model](threat-model.md).

## Everyday commands
| | |
|---|---|
| `membridge ask "..."` | what an assistant would be told for that question |
| `membridge serve` | the dashboard (prints a private link) |
| `membridge forget <id>` | erase a memory for real (assistants can only *ask*) |
| `membridge backup --encrypt` / `lock` | encrypted backup / encrypt the database at rest |
| `membridge audit-verify` | check nobody tampered with the access log |

## If something looks wrong
- *"Graph view: not running"*: fine. `docker compose up -d` (from a clone of the repo) if you want the graph tab.
- *Port busy*: `membridge serve --port 8800`.
- *It says it has nothing*: that is the point; ask something it was told, or import more.
- *Start over*: delete the data folder (`~/.membridge`, or `~/.membridge-demo`).
