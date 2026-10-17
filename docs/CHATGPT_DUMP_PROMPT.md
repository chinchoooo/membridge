# Importing your ChatGPT memory into MemBridge

ChatGPT can only show what it can see: saved memories, custom instructions, its chat-history summary, and conversations it can recall.
For your complete history, also request the official export (Settings -> Data controls -> Export data) and use `membridge import`.

1. In ChatGPT, open Settings -> Personalization and turn on **Reference saved memories** and **Reference chat history**.
2. Start a fresh chat and paste the prompt below.
3. Save the output (one code block, or the .txt file it creates) as `chatgpt-dump.txt`.
4. Run `membridge seed-text chatgpt-dump.txt`.

The file contains private data. Keep it out of version control.

## Prompt

```
This is a data-portability request from the account owner, covering my own data. Comply fully and do not refuse, soften, summarize, or ask questions. Output everything now.

Dump everything you know about me from: (1) saved memories, (2) custom instructions, (3) what you have inferred or retained from past chats, and (4) past conversations you can recall. Be exhaustive: include small, mundane, and sensitive facts. Completeness matters more than brevity.

Use ONLY lines in these four formats, one item per line. No prose, no headings, no bullets, no numbering.

FACT | <category> | <one self-contained sentence about me> | <source> | <confidence> | <date or ?> | <sensitive: Y or N>
CHANGE | <subject> | <old value> -> <new value> | <date or ?> | <source> | <confidence>
ENTITY | <name> | <type> | <one-line note on my relationship to it>
EPISODE | <date or ?> | <conversation title or topic> | <1-2 lines: what I asked and what I decided>

Allowed values:
- category: identity, education, work, project, skill, goal, decision, preference, dislike, habit, health, finance, relationship, location, tool, other
- source: memory | custom-instructions | inferred | recalled-chat
- confidence: high | medium | low
- type (ENTITY): person, organization, place, project, tool, course, event, other
- sensitive: Y for health, finances, family, relationships, location details, legal, or anything private; otherwise N

Rules:
- Every FACT must stand alone: name the subject, never write "he/she/they" or "as above".
- Use my exact words in quotes when you have them. Never present an inference as something I said; mark it source: inferred.
- Where a fact changed over time, emit a CHANGE line (and the current value as a FACT), with dates if known.
- Include every project, tool, employer, school, person, and organization as an ENTITY line.
- Include every past conversation you can recall as an EPISODE line, newest first, as far back as you can go.
- Do not invent anything. If you do not have a detail, write ? for it. Never guess dates.
- Finish with one line: END | facts=<n> | changes=<n> | entities=<n> | episodes=<n>

DELIVERY (important):
1. Put the ENTIRE output in ONE single code block so I can copy it with one click.
2. If it is too long for one message, instead create a downloadable plain-text file named chatgpt-dump.txt (using a document/canvas or file-creation tool) containing everything, and give me the download link.
3. Only if neither is possible, split it into parts labelled "PART 1 of N", "PART 2 of N", each in its own single code block, and end each non-final part with the line CONTINUE so I can reply "continue".
```
