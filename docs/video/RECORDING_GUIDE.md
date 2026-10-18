# Recording guide (about 30 minutes of your time)

You record two things. I do everything else (visuals, dashboard recordings, captions, assembly, blur, thumbnail).

| You record | File to give me | Script |
|---|---|---|
| **A. Voice-over** for scenes 1, 2, 4, 5, 6 | `submission/inbox/voiceover.wav` (or .mp3/.m4a) | `SCRIPT.md` / `teleprompter.html` |
| **B. Live screen recording** of scene 3 (ChatGPT + terminal), with your voice | `submission/inbox/live_chatgpt.mp4` (or .mkv) | section B below |

Then tell me the two paths and I assemble the video.

---

## A. Voice-over (10 minutes)
1. Quiet room, phone or laptop mic about a hand-span from your mouth. Windows: Voice Recorder or Audacity (free). No effects, no normalising.
2. Open `docs/video/teleprompter.html`, press space, and read scenes **1, 2, 4, 5, 6** in one go. **Skip scene 3** (just stay silent for its length, or stop and restart: either works).
3. After every scene: **two full seconds of silence**. That is how I find the cut points automatically. If you mess up, say the sentence again; I cut the bad take (I can only do that if you pause 2 s after it too).
4. Export WAV (or MP3 at 192 kbps+). Do not trim it.

## B. The live ChatGPT segment (15 minutes)

### Read this first: what will be on screen
ChatGPT's reply is a list of **everything it remembers about you**. In an earlier test that included health, money, family and location details. **If you upload this to public YouTube unmodified, those become public.** Pick one:

1. **Default (recommended): I blur the reply.** You record exactly as below; in assembly I blur the part of the screen where ChatGPT's reply scrolls and add a small caption "reply blurred: personal". You then watch the finished video once, end to end, before uploading.
2. **A demo ChatGPT account** with a few harmless memories ("I work at Acme Robotics", "I like Neovim"). Nothing to blur, but not "my own account".
3. Not recommended: Temporary Chat. It has no memory, so the reply is empty.

Whatever you choose: **do not scroll through or hover over the reply while recording**, and delete `chatgpt-dump.txt` and the video data folder afterwards.

### Prepare (off camera)
Open PowerShell and run:
```powershell
uv tool install --force git+https://github.com/chinchoooo/membridge     # installs the `membridge` command
$env:MEMBRIDGE_DATA_DIR = "$HOME\membridge-video"                      # a SEPARATE store; your real ~/.membridge is never touched
Remove-Item -Recurse -Force $env:MEMBRIDGE_DATA_DIR -ErrorAction SilentlyContinue
```
Do one **dry run** of the steps below so you know your first question works (pick something harmless that ChatGPT really knows and is not private, for example which internship you were selected for). Then run the `Remove-Item` line again so the real take starts clean.
Make the PowerShell font large (Ctrl + mouse wheel, or Settings > Defaults > Font size 20+), put the terminal on the left and the browser on the right, 1920x1080.

### The take
Start recording (OBS Display Capture 1920x1080 / 30 fps, or Win+G Xbox Game Bar). Talk as you work, following scene 3 of the script.

1. **Terminal:** `membridge dump-prompt | Set-Clipboard`  (this puts the prompt on your clipboard: say "I run dump-prompt").
2. **Browser, chatgpt.com, your account, a normal new chat:** click the box, **Ctrl+V**, **Enter**. It takes 30 to 90 seconds to answer. Say "I'll speed this part up" (I speed it up and blur it).
3. When it finishes, click the **Copy** button on the code block in the reply (top right of the block). **Do not scroll.**
4. **Terminal:** save and import it:
   ```powershell
   Get-Clipboard | Set-Content chatgpt-dump.txt -Encoding utf8
   membridge seed-text chatgpt-dump.txt
   ```
   You will see counts, including how many facts were marked `sensitive`. Say that those are hidden automatically.
5. **The question that works** (your harmless one):
   ```powershell
   membridge ask "Which internship was I selected for?" --short
   ```
6. **The question that fails, honestly:**
   ```powershell
   membridge ask "What is my favourite pizza topping?" --short
   ```
   Pause on the `[nothing stored]` line for two seconds. Say "that no is the feature".
7. Stop recording. Stay silent for 3 seconds first.

### Afterwards
```powershell
Remove-Item chatgpt-dump.txt
Remove-Item -Recurse -Force $env:MEMBRIDGE_DATA_DIR
```
Save the recording as `submission/inbox/live_chatgpt.mp4` (OBS: File > Remux Recordings to get an mp4).

## C. Give me the files
Put both in `submission/inbox/` and tell me. I produce `submission/final/membridge_demo.mp4`, `membridge_demo.srt` (captions), and a thumbnail, and I tell you which seconds of the ChatGPT reply were blurred so you can check them.

## D. Uploading to YouTube (you, 5 minutes)
studio.youtube.com > Create > Upload video > pick `membridge_demo.mp4`. Title and description are in `docs/SUBMISSION_FORM.md`. Upload the `.srt` under Subtitles. Visibility: **Unlisted** or **Public** (the form requires one of these). Copy the link and paste it into the form; send it to me and I put it in the README too.
