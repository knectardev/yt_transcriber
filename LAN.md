# LAN access

The transcription app runs on this Windows PC. Other computers and phones on the same network use a browser. Downloads, transcription, and project files stay on this PC. Nothing in this setup is a second server, a backup service, or a public website.

## Architecture

One Python process serves two ports:

- **8765** is the Streamlit app: the library, YouTube and MIDI forms, and the project editor.
- **8766** is the playback server in that same process: video, score, and fragment pages.

`baroque-transcribe ui` with no extra flags still listens only on `127.0.0.1`. LAN access is the start script below, which passes `--address 0.0.0.0`. Playback links use the hostname the browser used, and only when that host is this PC, its hostname, or a name listed in `BAROQUE_ALLOWED_HOSTS`.

Streamlit keeps its normal CORS and XSRF checks. The playback server answers only for tokens it issued, only for a fixed list of page names, and only for files inside the registered project or the preview cache. A browser that posts a sync offset must send an Origin for that same playback host.

## Start and stop

A desktop shortcut named **Piano transcription** runs `scripts\start-lan.ps1`. A scheduled task with the same name runs that script when you sign in, including after a reboot. The task is for the current user. It does not start before the sign-in screen, and it is not a Windows service.

If the server is already running, the shortcut says so and does not start a second copy. Closing that message window does not stop the server. Stop a server you started yourself with Ctrl+C in its window. Stop the sign-in copy by signing out, or by ending that Python process.

The same text is appended to `.local\lan-ui.log`. From a terminal, the command is:

```text
powershell -ExecutionPolicy Bypass -File scripts\start-lan.ps1
```

On this PC only, the old command is unchanged:

```text
baroque-transcribe ui
```

That listens on `http://127.0.0.1:8765` and does not accept LAN browsers.

## URLs

- This PC: `http://127.0.0.1:8765`
- Another device on the LAN: `http://192.168.40.137:8765`

The score and video pages use port **8766** on the same host the browser used. Those addresses are created when a project is opened. They change when the app process restarts, so bookmark the library URL, not a score URL.

## Firewall

Not applied yet. The Ethernet network **Dwelling** is currently **Public**. Public profile inbound connections are blocked unless a rule allows them.

Preferred rule, which leaves the network profile Public:

```text
New-NetFirewallRule -DisplayName "Baroque transcriber LAN" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8765,8766 -Profile Public -RemoteAddress LocalSubnet
```

`LocalSubnet` is the subnet of the interface that receives the connection (`192.168.40.0/24` on Ethernet). The rule does not forward a port on the router and does not allow the public internet. Reclassifying Dwelling from Public to Private is not required, and it would turn on every other Private-profile rule on this PC. Do not run the command above until it is approved.

An existing Windows rule allows `C:\users\chris\appdata\local\programs\python\python312\python.exe` on Public for any port and any remote address. The app's virtual environment is a different program path. This deployment does not use that rule and does not remove it.

## This network

Recorded for a later DHCP reservation. Nothing here was changed.

| Item | Value |
| --- | --- |
| Hostname | `Knectar_Alien` (FQDN `Knectar_Alien.lan`) |
| Interface | Ethernet, Realtek Gaming 2.5GbE |
| MAC | `CC-96-E5-00-7D-1B` |
| IPv4 | `192.168.40.137/24`, DHCP, gateway `192.168.40.1` |
| Profile | Public, name Dwelling |
| App ports | 8765 Streamlit, 8766 playback |

Wi-Fi is up with only a link-local address. It is not the path other devices should use.

## Project data

All of this stays where it already is:

- Projects, audio, MIDI, and MusicXML: `transcriptions\`
- Parse history: `data\parses`
- Fragments: `data\fragments`
- Search index and generated pages: `.local\`
- Model checkpoint: `%USERPROFILE%\.cache\baroque_transcribe\`

A transcription job writes a `.transcribe-incomplete` marker in its new folder. The library hides that folder until the job finishes. If the job fails, that folder is deleted. If the process is killed, the marked folder is left in place and is not merged into another project. Restarting the app does not resume the job.

A path typed into "Local file", "MIDI path", or the ZIP field is a path on this PC. The buttons labeled as downloads send a file to the browser. "Write ZIP on the server" writes only on this PC.

## Browsers

Windows Chrome, macOS browsers, and iPhone or iPad Safari can open the app over plain HTTP. YouTube embeds, the score, and audio playback do not need HTTPS. Safari may wait for a tap before audio starts.

Highlight and voice-color checkboxes are stored in that browser's `localStorage`. They are display preferences, not project data. The app does not use Web MIDI or the microphone. Those browser APIs would need HTTPS; they are not part of this app.

There is no login. Anyone who can open the LAN address can use the library.

## Later DNS

Do not change the router yet. When a name such as `transcriber.home.arpa` exists:

1. Reserve `CC-96-E5-00-7D-1B` for a stable address on `192.168.40.0/24`.
2. Point the name at that address.
3. Set `BAROQUE_ALLOWED_HOSTS` to that name before starting the app, then restart.

No source change is required. Until then, use `http://192.168.40.137:8765`.

## Rollback

1. Stop the server with Ctrl+C, or sign out if the sign-in task started it.
2. Remove the sign-in task with `Unregister-ScheduledTask -TaskName "Piano transcription" -Confirm:$false`.
3. Delete the desktop shortcut named Piano transcription.
4. Restore the code with git if these changes should be undone.
5. If the firewall rule was added later, remove it with `Remove-NetFirewallRule -DisplayName "Baroque transcriber LAN"`.
6. The network profile was not changed.

## Manual check from another device

After the firewall rule is approved and added:

1. On this PC, run `scripts\start-lan.ps1` and open `http://127.0.0.1:8765`.
2. On a laptop or phone, open `http://192.168.40.137:8765`.
3. Confirm a project card's score link uses `192.168.40.137` and port `8766`, then open it.
4. Submit a short YouTube URL. The download and the new project folder appear under `transcriptions\` on this PC. Close the laptop browser and confirm the job is still listed when you reconnect.
5. Upload a MIDI file from the other device and open the project.
6. On Edit & files, use a download button and confirm the file arrives on the other device. Use "Write ZIP on the server" with a path on this PC and confirm the ZIP is created here and is not offered as a browser download.
