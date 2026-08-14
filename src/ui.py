"""Presentation layer — animated components for the Streamlit app.

Streamlit's stock widgets are functional but flat. Everything here is plain CSS
and inline SVG injected through `unsafe_allow_html`, so it costs no extra
dependency and no JavaScript.

Two rules kept throughout:

* **Colours come from tokens, never hard-coded per component**, and every one is
  defined for both light and dark. Streamlit exposes the active theme through
  `prefers-color-scheme`, and a card that only looks right on one of them looks
  broken on the other.
* **Animation carries meaning.** Entrances are staggered in reading order so the
  eye lands on the verdict first; the confidence ring sweeps to its value so the
  number is felt as a proportion; a positive finding pulses once rather than
  continuously, because a clinical result that throbs forever reads as an alarm
  and gets ignored.
"""
from __future__ import annotations

import html

CSS = """
<style>
:root{
  --md-accent:#2E9E8F;      --md-accent-2:#4A9EDA;
  --md-danger:#E5534B;      --md-ok:#2CC985;
  --md-surface:rgba(130,140,160,.08);
  --md-surface-2:rgba(130,140,160,.14);
  --md-border:rgba(130,140,160,.22);
  --md-text-dim:rgba(130,140,160,.95);
  --md-shadow:0 6px 22px rgba(0,0,0,.10);
}
@media (prefers-color-scheme: dark){
  :root{
    --md-surface:rgba(255,255,255,.05);
    --md-surface-2:rgba(255,255,255,.09);
    --md-border:rgba(255,255,255,.14);
    --md-shadow:0 6px 26px rgba(0,0,0,.42);
  }
}

@keyframes mdFadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes mdFadeIn{from{opacity:0}to{opacity:1}}
/* These two animate TO a per-element value, which needs care on two counts.
   1. `animation-fill-mode: both` with only a `from` keyframe treats that single
      keyframe as first AND last, so the forwards fill pins the element at its
      start value forever - bars sat at zero and the ring drew nothing.
   2. `var()` inside @keyframes is not honoured by every engine (verified failing
      here), so the target cannot be expressed as a custom property either.
   The fix uses `backwards`: the first keyframe applies during the delay, and
   once the animation ends the element simply reverts to its own inline value.
   No var() in a keyframe, no forwards fill pinning it. */
@keyframes mdGrow{from{transform:scaleX(0)}}
@keyframes mdSweep{from{stroke-dashoffset:340}}   /* 340 > 2*pi*54 = 339.29 */
@keyframes mdPulse{
  0%{box-shadow:0 0 0 0 rgba(229,83,75,.42)}
  70%{box-shadow:0 0 0 16px rgba(229,83,75,0)}
  100%{box-shadow:0 0 0 0 rgba(229,83,75,0)}
}
@keyframes mdDrift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes mdShimmer{100%{transform:translateX(100%)}}

/* Respect a reduced-motion preference, and make it structurally safe: with
   animations off, every element falls back to its own inline value - the bar to
   its real width, the ring to its real arc - so nothing can be left stranded at
   an animation's starting state. */
@media (prefers-reduced-motion: reduce){
  *{animation:none !important;transition:none !important}
}

/* ---------------------------------------------------------------- hero */
.md-hero{
  position:relative;overflow:hidden;border-radius:18px;padding:26px 30px;
  margin:2px 0 20px;border:1px solid var(--md-border);
  background:linear-gradient(120deg,rgba(46,158,143,.20),rgba(74,158,218,.16),
             rgba(126,109,214,.18),rgba(46,158,143,.20));
  background-size:300% 300%;animation:mdDrift 18s ease infinite,mdFadeIn .5s ease both;
}
.md-hero h1{margin:0;font-size:1.85rem;font-weight:700;letter-spacing:-.4px;line-height:1.15}
.md-hero p{margin:8px 0 0;font-size:.95rem;opacity:.78;max-width:62ch;line-height:1.5}
.md-hero .md-badge{
  display:inline-block;margin-top:14px;padding:5px 13px;border-radius:999px;
  font-size:.74rem;font-weight:600;letter-spacing:.4px;text-transform:uppercase;
  background:var(--md-surface-2);border:1px solid var(--md-border);
}

/* --------------------------------------------------------------- stats */
.md-stats{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 20px}
.md-stat{
  flex:1 1 150px;padding:16px 18px;border-radius:14px;
  background:var(--md-surface);border:1px solid var(--md-border);
  animation:mdFadeUp .5s cubic-bezier(.22,1,.36,1) both;
  transition:transform .22s cubic-bezier(.22,1,.36,1),box-shadow .22s,border-color .22s;
}
.md-stat:hover{transform:translateY(-4px);box-shadow:var(--md-shadow);
  border-color:rgba(46,158,143,.55)}
.md-stat .md-k{font-size:.7rem;text-transform:uppercase;letter-spacing:.7px;
  opacity:.62;font-weight:600}
.md-stat .md-v{font-size:1.7rem;font-weight:700;margin-top:5px;line-height:1.1;
  background:linear-gradient(90deg,var(--md-accent),var(--md-accent-2));
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.md-stat .md-s{font-size:.72rem;opacity:.55;margin-top:3px}

/* ------------------------------------------------------------- verdict */
.md-verdict{
  display:flex;gap:26px;align-items:center;flex-wrap:wrap;
  padding:24px 26px;border-radius:18px;margin:4px 0 18px;
  background:var(--md-surface);border:1px solid var(--md-border);
  animation:mdFadeUp .55s cubic-bezier(.22,1,.36,1) both;
}
.md-verdict.pos{border-color:rgba(229,83,75,.5);animation:mdFadeUp .55s cubic-bezier(.22,1,.36,1) both,mdPulse 1.8s ease-out .5s 1}
.md-verdict.neg{border-color:rgba(44,201,133,.45)}
.md-ring{flex:0 0 auto;position:relative;width:132px;height:132px}
.md-ring svg{transform:rotate(-90deg)}
.md-ring .md-track{stroke:var(--md-surface-2)}
.md-ring .md-fill{
  stroke-linecap:round;stroke-dasharray:339.29;
  animation:mdSweep 1.15s cubic-bezier(.22,1,.36,1) backwards;
}
.md-ring .md-num{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;animation:mdFadeIn .8s .35s both;
}
.md-ring .md-num b{font-size:1.62rem;font-weight:700;line-height:1}
.md-ring .md-num span{font-size:.64rem;opacity:.6;text-transform:uppercase;
  letter-spacing:.6px;margin-top:3px}
.md-vtext{flex:1 1 240px;min-width:0}
.md-vtext .md-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;
  opacity:.6;font-weight:600}
.md-vtext h2{margin:6px 0 10px;font-size:1.65rem;font-weight:700;line-height:1.2}
.md-vtext h2.pos{color:var(--md-danger)} .md-vtext h2.neg{color:var(--md-ok)}
.md-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px}
.md-chip{
  padding:5px 12px;border-radius:999px;font-size:.76rem;font-weight:600;
  background:var(--md-surface-2);border:1px solid var(--md-border);
  animation:mdFadeUp .45s cubic-bezier(.22,1,.36,1) both;
}

/* ------------------------------------------------------------ vote rows */
.md-votes{margin:6px 0 4px}
.md-vote{
  display:grid;grid-template-columns:1fr auto;gap:4px 12px;align-items:center;
  padding:10px 14px;margin-bottom:7px;border-radius:11px;
  background:var(--md-surface);border:1px solid var(--md-border);
  animation:mdFadeUp .45s cubic-bezier(.22,1,.36,1) both;
  transition:transform .18s,border-color .18s;
}
.md-vote:hover{transform:translateX(4px);border-color:rgba(74,158,218,.5)}
.md-vote .md-name{font-weight:600;font-size:.9rem}
.md-vote .md-fam{font-size:.7rem;opacity:.5;margin-left:7px;font-weight:400}
.md-vote .md-pct{font-variant-numeric:tabular-nums;font-weight:700;font-size:.92rem}
.md-bar{grid-column:1/-1;height:6px;border-radius:99px;background:var(--md-surface-2);
  overflow:hidden;margin-top:2px}
.md-bar i{display:block;height:100%;width:100%;border-radius:99px;
  transform-origin:left center;
  animation:mdGrow .9s cubic-bezier(.22,1,.36,1) backwards}
.md-legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.74rem;opacity:.62;
  margin:10px 2px 2px}
.md-legend b{font-weight:600;opacity:.85}

/* ------------------------------------------------------------- sundry */
.md-note{
  padding:13px 16px;border-radius:12px;font-size:.84rem;line-height:1.55;
  background:var(--md-surface);border-left:3px solid var(--md-accent);
  animation:mdFadeIn .5s both;
}
div[data-testid="stMetric"]{
  background:var(--md-surface);border:1px solid var(--md-border);
  border-radius:13px;padding:13px 15px;
  transition:transform .2s cubic-bezier(.22,1,.36,1),box-shadow .2s;
}
div[data-testid="stMetric"]:hover{transform:translateY(-3px);box-shadow:var(--md-shadow)}
.stButton>button{
  border-radius:11px;font-weight:600;letter-spacing:.2px;
  transition:transform .18s cubic-bezier(.22,1,.36,1),box-shadow .18s,filter .18s;
}
.stButton>button:hover{transform:translateY(-2px);box-shadow:var(--md-shadow);filter:brightness(1.06)}
.stButton>button:active{transform:translateY(0)}
.stTabs [data-baseweb="tab"]{transition:color .18s,background .18s;border-radius:9px 9px 0 0}
img{border-radius:12px}
</style>
"""


def inject() -> None:
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def _esc(s) -> str:
    return html.escape(str(s))


def hero(title: str, subtitle: str = "", badge: str = "") -> str:
    parts = [f'<div class="md-hero"><h1>{_esc(title)}</h1>']
    if subtitle:
        parts.append(f"<p>{_esc(subtitle)}</p>")
    if badge:
        parts.append(f'<span class="md-badge">{_esc(badge)}</span>')
    parts.append("</div>")
    return "".join(parts)


def stats(items) -> str:
    """items: iterable of (label, value, sub). Entrances stagger left to right."""
    cells = []
    for i, item in enumerate(items):
        label, value = item[0], item[1]
        sub = item[2] if len(item) > 2 else ""
        cells.append(
            f'<div class="md-stat" style="animation-delay:{0.05 * i:.2f}s">'
            f'<div class="md-k">{_esc(label)}</div>'
            f'<div class="md-v">{_esc(value)}</div>'
            + (f'<div class="md-s">{_esc(sub)}</div>' if sub else "")
            + "</div>")
    return f'<div class="md-stats">{"".join(cells)}</div>'


def verdict(title: str, probability: float, positive: bool,
            chips=(), caption: str = "") -> str:
    """Big result card with a ring that sweeps to the probability."""
    p = max(0.0, min(1.0, float(probability)))
    r = 54
    circ = 2 * 3.141592653589793 * r
    offset = circ * (1 - p)
    colour = "var(--md-danger)" if positive else "var(--md-ok)"
    cls = "pos" if positive else "neg"

    chip_html = "".join(
        f'<span class="md-chip" style="animation-delay:{0.35 + 0.07 * i:.2f}s">'
        f"{_esc(c)}</span>" for i, c in enumerate(chips))

    return f"""
<div class="md-verdict {cls}">
  <div class="md-ring">
    <svg width="132" height="132" viewBox="0 0 132 132">
      <circle class="md-track" cx="66" cy="66" r="{r}" fill="none" stroke-width="11"/>
      <circle class="md-fill" cx="66" cy="66" r="{r}" fill="none" stroke-width="11"
              stroke="{colour}" style="stroke-dashoffset:{offset:.2f}"/>
    </svg>
    <div class="md-num"><b>{p * 100:.1f}%</b><span>probability</span></div>
  </div>
  <div class="md-vtext">
    <div class="md-label">Final weighted verdict</div>
    <h2 class="{cls}">{_esc(title)}</h2>
    <div class="md-chips">{chip_html}</div>
    {f'<p style="margin:12px 0 0;font-size:.84rem;opacity:.68;line-height:1.5">{_esc(caption)}</p>' if caption else ''}
  </div>
</div>"""


def votes(rows, positive_label: str, negative_label: str,
          agree_with: str | None = None) -> str:
    """rows: dicts with Algorithm, Family, Vote, Surety %, Weight %.

    Bars are coloured by which way the model voted and widths animate in, so
    agreement and disagreement are visible before any number is read.

    `agree_with` switches to the multi-class reading used for scans, where
    "positive" is not a single class: a bar is then red when that model
    disagrees with the final verdict, rather than when it voted for a disease.
    """
    out = ['<div class="md-votes">']
    for i, r in enumerate(rows):
        vote = str(r.get("Vote", ""))
        if agree_with is not None:
            voted_pos = vote != agree_with        # red == dissenting
        else:
            voted_pos = vote == positive_label
        colour = ("linear-gradient(90deg,#E5534B,#F08A72)" if voted_pos
                  else "linear-gradient(90deg,#2CC985,#4A9EDA)")
        weight = float(r.get("Weight %", 0.0))
        surety = float(r.get("Surety %", 0.0))
        delay = 0.05 * i
        out.append(
            f'<div class="md-vote" style="animation-delay:{delay:.2f}s">'
            f'<div><span class="md-name">{_esc(r["Algorithm"])}</span>'
            f'<span class="md-fam">{_esc(r.get("Family", ""))}</span></div>'
            f'<div class="md-pct" style="color:{"var(--md-danger)" if voted_pos else "var(--md-ok)"}">'
            f'{_esc(r.get("Vote", ""))} · {surety:.0f}%</div>'
            # scaleX rather than width: the target is written as a literal, so
            # nothing depends on var() resolving inside a keyframe
            f'<div class="md-bar"><i style="transform:scaleX({max(surety, 2) / 100:.4f});'
            f'background:{colour};animation-delay:{delay + 0.1:.2f}s"></i></div>'
            f'</div>')
    out.append("</div>")
    if agree_with is not None:
        legend = (f'<span><b>Green</b> = agrees with the verdict '
                  f'({_esc(agree_with)})</span>'
                  f'<span><b>Red</b> = dissenting</span>')
    else:
        legend = (f'<span><b>Red</b> = voted {_esc(positive_label)}</span>'
                  f'<span><b>Green</b> = voted {_esc(negative_label)}</span>')
    out.append('<div class="md-legend">'
               "<span><b>Bar</b> = that model's surety in this case</span>"
               f"{legend}</div>")
    return "".join(out)


def note(text: str) -> str:
    return f'<div class="md-note">{text}</div>'
