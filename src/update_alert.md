## **✨ Major Update Alert: Text-Based Style Prompts & Wild Instrumental Generation — Unleash Your Musical Madness!**

Hey DiffRhythm community,

Last week’s release of **DiffRhythm** hid two *secret features*—and you brilliant creators found them all! Thanks to sharp-eyed explorers in our community (shoutout to @Jourdelune for his epic code contributions!), we’re thrilled to officially announce: **Text-to-Style prompts** and **Instrumental Music Generation** are now live in the open-source code!

### 🎨 Craft Styles with Words

Describe your vision *literally* — from quick tags to cinematic scenes. DiffRhythm speaks your language:

**Straightforward prompts**: 

- `Pop Emotional Piano` [📀 Demo Track](https://aslp-lab.github.io/DiffRhythm.github.io/raw/samples/syn/llm_demo7_classic_last_1.wav)

(Vocal glitches acknowledged → Can't resist sharing this gorgeous instrumental track!)

- `Jazzy Nightclub Vibe` [📀 Demo Track](https://aslp-lab.github.io/DiffRhythm.github.io/raw/samples/syn/llm_demo7_classic_last_1.wav)

etc.

**Vivid storytelling**:

- `Indie folk ballad, coming-of-age themes, acoustic guitar picking with harmonica interludes` 
[📀 Demo Track](https://aslp-lab.github.io/DiffRhythm.github.io/raw/samples/syn/llm_demo7_classic_last_1.wav)

- `Buenos Aires tango-rap graffiti artist's rhymes over bandoneón syncopation` 
[📀 Demo Track](https://aslp-lab.github.io/DiffRhythm.github.io/raw/samples/syn/llm_demo7_classic_last_1.wav)

**No reference audio needed**—type your imagination, and let Diffrhythm handle the rest.

### 🎻 Instrumental Mode: Lyrics Optional, Chaos Mandatory

Skip the lyrics and go wild with *absurdly specific prompts*:

- `Mountain cabin fireplace, acoustic guitar folk tunes wrapped in wool blankets`
  [📀 Demo Track](https://aslp-lab.github.io/DiffRhythm.github.io/raw/samples/syn/llm_demo7_classic_last_1.wav)
- `Arctic research station, theremin auroras dancing with geomagnetic storms`
  [📀 Demo Track](https://aslp-lab.github.io/DiffRhythm.github.io/raw/samples/syn/llm_demo7_classic_last_1.wav)

 **or** merge them with wild prompts – it’s your playground：

- `Zombie apocalypse country-rock gas station escape, banjo shreds & shotgun reload beats`
  [📀 Demo Track](https://aslp-lab.github.io/DiffRhythm.github.io/raw/samples/syn/llm_demo7_classic_last_1.wav)
- `AI love rival duet: human vs vocal assistant jealousy battle, vocoder vs raw screams`
   (⚠️ Audio Safety Notice: This demo may redefine "loud".🎧 Lower volume → 😌 | Keep volume → 🤯 Your choice, brave listener!)
  [📀 Demo Track](https://aslp-lab.github.io/DiffRhythm.github.io/raw/samples/syn/llm_demo7_classic_last_1.wav)

**Rules? We deleted them.** Throw your weirdest ideas at DiffRhythm and see what sticks.

### 🌟 The Power of Open-Source Alchemy

This update began as a *community-powered quest*:

- Developer @Jourdelune discovered our hidden API endpoints while tinkering with the code
- They not only implemented the features but submitted a clean, elegant Pull Request
- **Now their code lives in the main repo**—this is open-source magic at its finest!

### 🚀 Get Started Now

- **GitHub Repo**: [https://github.com/ASLP-lab/DiffRhythm] 
- **Huggingface Space Demo**: [https://huggingface.co/spaces/ASLP-lab/DiffRhythm] 

**We dare you to:** 👉 Share your “this-shouldn’t-work-but-it-does” creations in Community 👉 Join the #DiffRhythm channel on [Discord] to witness collective genius/insanity 👉 Star the repo or become the next code wizard!

### 🔧 **Community Feedback Drives Progress**

A massive shoutout to @corybrsn and everyone who shared sharp insights on current limitations. Here’s our deep dive into your questions — **transparency is key**!

- **Musicality Improvements** 🎵 *"Sometimes the musicality feels uncanny or off-key."* **Our Response**: We agree! This stems from limitations in harmony/melody modeling. **Next steps**: Training a larger model with higher-quality, genre-diverse data. Think "less robotic, more Radiohead" vibes.
- **Lyric Consistency** 📝 *"Words get skipped or repeated occasionally."* **Root Cause**: The timestamp constraint is a double-edged sword. If the time between lines is too short → words get squeezed; too long → repetition kicks in. **Fixes in Progress**: We’re rebalancing timestamp flexibility (less rigidity!) and testing adaptive duration prediction.
- **Structured Lyrics** 🕒 *"Can we label [verse]/[chorus] sections?"* **Plan**: Absolutely! Inspired by tools like Suno, we’re adding **lyric structure tags** (e.g., `[intro]`, `[bridge]`) for granular control. Bonus: This will simplify timestamp input — describe sections, not milliseconds!

### 🤖 **Why This Matters**

Your critiques aren’t just "issues" — they’re **blueprints for evolution**. We’re tackling these challenges head-on because:

1. Better musicality = More professional-grade outputs
2. Smarter timestamp handling = Less engineering, more creating
3. Structured lyrics = Democratizing music production

**To All Sound Rebels:** You’re why DiffRhythm evolved from a “tool” to a **movement**. When we see tracks like *Quantum Hotpot Folk* or *Sad Washing Machine Glitchcore*—we know **you’re rewriting the rules of music itself.**

Next-gen features are brewing (spoilers: song editing? vocal cloning?), but here’s the real ask: **Keep bringing the chaos. We’ll keep turning it into sound.**

**To Every Tester & Code Contributor**: You’re not just users — you’re **co-developers**. When you dissect the model’s quirks, we gain superpowers to rebuild it. **Keep the feedback flowing!**

— The ASLP-lab DiffRhythm Team & All Open-Source Alchemists

**P.S.** To everyone who commented “this project will be abused”… *You were right, and we’re proud.* 🚀