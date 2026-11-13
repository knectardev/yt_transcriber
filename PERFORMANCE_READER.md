# Performance reader (Vercel static PWA)

The desktop Python app remains authoritative. Vercel hosts only the **`reader/`** partition: a static score reader plus a published, read-only Bach catalog projection.

## Vercel project settings

Configure the linked GitHub project (`knectardev/yt_transcriber`) as follows:

| Setting | Value |
| --- | --- |
| Framework Preset | **Other** (not Python) |
| Root Directory | empty / repository root **or** `reader` |
| Install / Build / Output | Prefer repo-root [`vercel.json`](vercel.json): no-op install/build, `outputDirectory: "reader"` |

If Root Directory is set to `reader`, then Output Directory is `.` and [`reader/vercel.json`](reader/vercel.json) applies instead.

Vercel must **never** install or execute the desktop Python application. There is no Python entrypoint on purpose. If the project still shows a Python icon / `PYTHON_ENTRYPOINT_NOT_FOUND`, open **Settings → General → Framework Preset → Other**, save, then **Redeploy**.

## Publishing mechanism (v1)

**Git-tracked static output** deployed through the existing Vercel Git integration.

1. On the desktop: `baroque-transcribe reader publish --id bwv-0772 …`
2. That writes/merges packages under `reader/catalog/` and refreshes shell assets (including OSMD).
3. Commit and push `reader/` changes.
4. Vercel rebuilds from Root Directory `reader` only.

This is **not** Vercel CLI deploy for v1. Do not gitignore `reader/catalog/` — published JSON packages are the public projection and must be tracked.

Incremental publish **preserves** previously published entries unless you run `reader remove`.

## CLI

```text
baroque-transcribe reader publish --id bwv-0772 --id bwv-0773
baroque-transcribe reader publish --id bwv-0772
baroque-transcribe reader remove --id bwv-0773
baroque-transcribe reader list
```

The same operations are available as Python functions in `baroque_transcribe.reader_publish` for a future desktop UI.

## Local preview

Serve the partition without Python app code:

```text
python -m http.server 4173 --directory reader
```

Open `http://127.0.0.1:4173/`.

## Shareable piece URLs

Each published score has a stable query link (no catalog search required):

```text
https://yt-transcriber-self.vercel.app/?id=bwv-0772
https://yt-transcriber-self.vercel.app/?id=bwv-0772&m=12
```

- `id` — catalog entry id (for example `bwv-0772`)
- `m` — optional printed measure number to restore reading position

Opening a piece from the catalog updates the address bar; browser Back returns to the catalog list.

## What is not on Vercel

Streamlit, playback server, transcription model, `transcriptions/`, LAN tooling, write APIs, and private audio.
