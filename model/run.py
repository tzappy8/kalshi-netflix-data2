"""
Netflix US #1 movie model, Stage 2 (v2).  python3 model/run.py  (run from the repo root)
Reads:  data/kalshi_log.csv, data/top10_log.csv, data/weekly_log.csv (from the laptop logger)
        model/*.csv|json (history, Stage 1 coefficients, title facts, seed daily lists)
Writes: model_output/report.html, model_output/latest.json

How it works
 1. Stage 1 prior: a conditional logit on 5 years of weekly charts gives each movie a score V1.
 2. Daily data: each daily Top 10 rank becomes estimated views (rank curve measured from Netflix's
    weekly view counts), weighted by day of week, projected forward with a launch/decay curve.
 3. Blend: each movie has a peak-day level. Stage 1 sets a prior on it; observed daily ranks update it
    (more days seen = more weight). Weekly views = level x the movie's days in the window, so a Friday
    release only counts Fri-Sun. Monte Carlo over the uncertainty gives P(#1).
 4. Flags: model vs. the Kalshi price, flagged only when the gap clears costs and enough data exists.
Assumptions to calibrate as weeks are logged: day-of-week weights, launch/decay curve, noise sizes.
"""
import json, math, os, sys, datetime as dt
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "model"); DATA = os.path.join(ROOT, "data"); OUT = os.path.join(ROOT, "model_output")
RANK_SHARE = {1: 1.0, 2: .579, 3: .371, 4: .287, 5: .238, 6: .21, 7: .183, 8: .16, 9: .146, 10: .131}  # median, Netflix weekly views by rank
OFF_LIST = 0.10                       # a released movie missing from the daily Top 10 (below #10)
DOW = [0.92, 0.88, 0.88, 0.92, 1.08, 1.20, 1.12]   # Mon..Sun viewing weights (assumption)
FIELD_P = 0.03                        # chance #1 is a movie with no Kalshi contract
EDGE_MIN, VOL_MIN, DATA_MIN = 0.10, 200.0, 0.30
N_SIM = 20000
rng = np.random.default_rng(7)

def age_curve(age_days, kind):
    """relative daily views by days since release (0 = release day)."""
    if kind == "old":                 # older licensed/catalog movie
        return 0.97 ** age_days
    if age_days < 0: return 0.0
    if age_days == 0: return 0.85
    if age_days <= 2: return 1.0
    a = age_days - 2
    return 0.88 ** min(a, 12) * 0.92 ** max(0, min(a - 12, 7)) * 0.97 ** max(0, a - 19)

def read_csv(p, **k):
    return pd.read_csv(p, **k) if os.path.exists(p) else None

# ---------- 1. figure out the live market and its week ----------
ks = [x for x in (read_csv(os.path.join(DATA, "kalshi_log.csv")), read_csv(os.path.join(DATA, "kalshi_prices.csv"))) if x is not None and not x.empty]
if not ks: sys.exit("no Kalshi data yet")
k = pd.concat(ks, ignore_index=True)       # kalshi_prices.csv (newer logger) also has order sizes
k["checked_at"] = pd.to_datetime(k.checked_at)
k = k[k.event.str.startswith("KXNETFLIXRANKMOVIE-")]
event = sorted(k.event.unique(), key=lambda e: dt.datetime.strptime(e.split("-")[1], "%y%b%d"))[-1]
close_day = dt.datetime.strptime(event.split("-")[1], "%y%b%d").date()      # e.g. 26NOV16 = Mon Nov 16
week_end = close_day - dt.timedelta(days=1)                                   # Sunday
week_start = week_end - dt.timedelta(days=6)
kk = k[k.event == event]
last_check = kk.checked_at.max()
px = kk[kk.checked_at >= last_check - pd.Timedelta(minutes=5)].copy()   # latest snapshot (both log files)
px["has_size"] = px["yes_ask_size"].notna() if "yes_ask_size" in px else False
px = px.sort_values(["checked_at", "has_size"]).drop_duplicates("title", keep="last")
for c in ("yes_bid", "yes_ask", "volume", "yes_bid_size", "yes_ask_size"):
    px[c] = pd.to_numeric(px[c], errors="coerce") if c in px else np.nan
px[["yes_bid", "yes_ask", "volume"]] = px[["yes_bid", "yes_ask", "volume"]].fillna(0)
px = px[(px.title.notna()) & (px.title != "DEACTIVATED")]

# ---------- 2. last week's official chart ----------
hist = pd.read_csv(os.path.join(M, "us_weekly_history.csv"), parse_dates=["week"])
wl = read_csv(os.path.join(DATA, "weekly_log.csv"))
if wl is not None and not wl.empty:
    wl = wl.rename(columns={"week_end": "week", "rank": "weekly_rank", "title": "show_title", "weeks": "cumulative_weeks_in_top_10"})
    wl["week"] = pd.to_datetime(wl.week)
    hist = pd.concat([hist, wl[hist.columns]]).drop_duplicates(["week", "weekly_rank"], keep="last")
prev_end = pd.Timestamp(week_start - dt.timedelta(days=1))
last = hist[hist.week == prev_end]
last_week_known = not last.empty
if not last_week_known:                 # fall back to the newest chart we have
    last = hist[hist.week == hist.week.max()]

# ---------- 3. daily lists ----------
frames = []
s = read_csv(os.path.join(M, "seed_daily.csv"))
if s is not None:
    s["view_day"] = pd.to_datetime(s.list_date) - pd.Timedelta(days=1)
    frames.append(s[["view_day", "rank", "title"]])
t = read_csv(os.path.join(DATA, "top10_log.csv"))
if t is not None and not t.empty:
    t["view_day"] = (pd.to_datetime(t.seen_at) - pd.Timedelta(hours=12)).dt.normalize() - pd.Timedelta(days=0)
    frames.append(t[["view_day", "rank", "title"]])
daily = pd.concat(frames) if frames else pd.DataFrame(columns=["view_day", "rank", "title"])
daily["view_day"] = pd.to_datetime(daily.view_day).dt.date
daily = daily.drop_duplicates(["view_day", "rank"], keep="last")
daily = daily[(daily.view_day >= week_start) & (daily.view_day <= week_end)]
days_obs = sorted(daily.view_day.unique())

# ---------- 4. candidates + Stage 1 prior ----------
cfg = json.load(open(os.path.join(M, "week_config.json"))).get("titles", {})
flags = pd.read_csv(os.path.join(M, "title_flags.csv")).set_index("title")
co = json.load(open(os.path.join(M, "stage1_coefs.json")))
norm = lambda x: str(x).replace("Tyler Perry's ", "").replace("’", "'").strip().lower()
names = {}
for n in list(last.show_title) + list(px.title) + list(daily.title) + list(cfg):
    names.setdefault(norm(n), n)
rows = []
for key, name in names.items():
    lw = last[last.show_title.map(norm) == key]
    c = next((v for kname, v in cfg.items() if norm(kname) == key), {})
    fl = flags.loc[name] if name in flags.index else None
    get = lambda f: int(c.get(f, (fl[f] if fl is not None and not pd.isna(fl[f]) else 0)))
    new = int(lw.empty)
    rel = pd.to_datetime(c["release"]).date() if c.get("release") else None
    if new and rel is None:
        seen = daily[daily.title.map(norm) == key].view_day
        rel = min(seen) if len(seen) else week_start
    if rel and rel < week_start: rel = week_start if new else rel
    days = (week_end - max(rel, week_start)).days + 1 if new else 7
    orig = get("original")
    x = dict(new=new, original=orig, orig_new=orig * new, orig_late=int(new and orig and days <= 3),
             ret_last_rank=0 if new else int(lw.weekly_rank.iloc[0]),
             ret_wk2=int(not new and lw.cumulative_weeks_in_top_10.iloc[0] == 1),
             ret_wk3p=int(not new and lw.cumulative_weeks_in_top_10.iloc[0] >= 2),
             last_was_1=int(not new and lw.weekly_rank.iloc[0] == 1),
             catalog=get("catalog"), non_english=get("non_english"), theatrical=get("theatrical"))
    v1 = float(np.dot((np.array([x[cn] for cn in co["cols"]], float) - co["mu"]) / co["sd"], co["beta"]))
    weeks_before = 0 if new else int(lw.cumulative_weeks_in_top_10.iloc[0])
    kind = "old" if (x["catalog"] and not new) or weeks_before >= 4 else "new"
    if new: age0 = (week_start - rel).days if rel else 0
    else: age0 = 7 * weeks_before + 3          # rough: released mid-week of its first chart week
    rows.append(dict(key=key, title=name, new=new, release=rel, days=days, kind=kind, age0=age0, V1=v1,
                     last_rank=None if new else int(lw.weekly_rank.iloc[0])))
C = pd.DataFrame(rows)

# ---------- 5. daily data -> projected weekly views ----------
window = [week_start + dt.timedelta(days=i) for i in range(7)]
def title_days(r):
    out = []
    for d in window:
        age = (d - r.release).days if (r.new and r.release) else r.age0 + (d - week_start).days
        if r.new and r.release and d < r.release: continue
        out.append((d, age, DOW[d.weekday()] * age_curve(age, r.kind)))
    return out
levels, wfrac, nobs, best_rank, shape = [], [], [], [], []
for r in C.itertuples():
    td = title_days(r); exp_all = sum(w for _, _, w in td) or 1e-9
    obs = [(d, a, w) for d, a, w in td if d in days_obs]
    lv = []
    for d, a, w in obs:
        hit = daily[(daily.view_day == d) & (daily.title.map(norm) == r.key)]
        v = (RANK_SHARE[int(hit["rank"].iloc[0])] if len(hit) else OFF_LIST) * DOW[d.weekday()]
        lv.append(v / max(w, 1e-9))                    # implied peak-day level from this day
    ranks = daily[daily.title.map(norm) == r.key]["rank"]
    best_rank.append(int(ranks.min()) if len(ranks) else None)
    levels.append(float(np.exp(np.average(np.log(lv), weights=np.arange(1, len(lv) + 1)))) if lv else np.nan)
    wfrac.append(sum(w for d, a, w in obs) / exp_all if lv else 0.0)
    nobs.append(len(obs)); shape.append(exp_all)
C["level_data"], C["w"], C["days_observed"], C["best_daily_rank"], C["window_weight"] = levels, wfrac, nobs, best_rank, shape

# ---------- 6. Monte Carlo ----------
# Each movie has a peak-day level L (1.0 = a typical daily #1). Weekly views = L x (its day weights in the window),
# so a Friday release only gets Fri-Sun. Prior on log L comes from Stage 1; daily data pulls it toward what was seen.
PRIOR_SD, DATA_SD = 0.9, 0.25
c0 = -np.sort(C.V1.values)[-1]                 # the strongest Stage 1 contender centred on level 1.0
prior_mu = c0 + C.V1.values
post_mu, post_sd = [], []
for i, r in enumerate(C.itertuples()):
    if np.isnan(r.level_data):
        post_mu.append(prior_mu[i]); post_sd.append(PRIOR_SD)
    else:
        sd_d = DATA_SD / math.sqrt(r.days_observed) + 0.35 * (1 - r.w)   # less certain when most of the week is still ahead
        wt = PRIOR_SD**-2 + sd_d**-2
        post_mu.append((prior_mu[i] / PRIOR_SD**2 + math.log(r.level_data) / sd_d**2) / wt); post_sd.append(wt ** -0.5)
C["post_mu"], C["post_sd"] = post_mu, post_sd
logL = C.post_mu.values + rng.normal(size=(N_SIM, len(C))) * C.post_sd.values
Wk = np.exp(logL) * C.window_weight.values
win = np.bincount(Wk.argmax(1), minlength=len(C)) / N_SIM
C["p_model"] = (1 - FIELD_P) * win
C["p_stage1"] = (1 - FIELD_P) * np.exp(C.V1 - C.V1.max()) / np.exp(C.V1 - C.V1.max()).sum()

# ---------- 7. compare with Kalshi + flags ----------
pm = px.assign(key=px.title.map(norm)).set_index("key")
C["bid"] = C.key.map(pm.yes_bid); C["ask"] = C.key.map(pm.yes_ask); C["volume"] = C.key.map(pm.volume)
C["bid_size"] = C.key.map(pm.yes_bid_size); C["ask_size"] = C.key.map(pm.yes_ask_size)
C["mid"] = (C.bid + C.ask) / 2
C["edge_yes"] = C.p_model - C.ask; C["edge_no"] = C.bid - C.p_model
def flag(r):
    if pd.isna(r.ask): return ""
    side, edge = ("Yes", r.edge_yes) if r.edge_yes >= r.edge_no else ("No", r.edge_no)
    if edge < EDGE_MIN: return ""
    if (r.volume or 0) < VOL_MIN: return ""
    if r.w < DATA_MIN: return ""
    why = (f"{r.days_observed} day(s) of daily data, best daily rank #{int(r.best_daily_rank)}" if r.days_observed else "no daily data")
    size = r.ask_size if side == "Yes" else r.bid_size
    thin = "" if pd.isna(size) else (f" Thin: only {size:.0f} contract{'s' if size != 1 else ''} at that price." if size < 20 else f" {size:.0f} contracts at that price.")
    price = r.ask if side == "Yes" else 1 - r.bid
    return f"Buy {side} on {r.title}: model {r.p_model if side=='Yes' else 1-r.p_model:.0%} vs {price*100:.0f}\u00a2 to buy ({edge*100:.0f} pts). {why}.{thin}"
C["flag"] = C.apply(flag, axis=1)
C = C.sort_values("p_model", ascending=False)

# ---------- 8. write outputs ----------
os.makedirs(OUT, exist_ok=True)
from zoneinfo import ZoneInfo
run_at = dt.datetime.now(ZoneInfo("America/New_York")).strftime("%b %d %I:%M %p")
meta = dict(run_at=run_at, event=event, week=f"{week_start} to {week_end}", kalshi_prices_at=str(last_check),
            daily_days=[str(d) for d in days_obs], last_week_chart_known=last_week_known, field_p=FIELD_P)
keep = ["title", "new", "release", "days", "days_observed", "best_daily_rank", "w", "p_stage1", "p_model", "bid", "ask", "bid_size", "ask_size", "mid", "volume", "edge_yes", "edge_no", "flag"]
json.dump(dict(meta=meta, rows=json.loads(C[keep].to_json(orient="records", date_format="iso"))),
          open(os.path.join(OUT, "latest.json"), "w"), indent=1, default=str)
def pct(x): return "" if pd.isna(x) else f"{x:.0%}"
def cents(x): return "" if pd.isna(x) else f"{x*100:.0f}¢"
def gap(r):
    if pd.isna(r.mid): return ""
    g = (r.p_model - r.mid) * 100
    return f"{g:+.0f}"
trs = "".join(
    f"<tr class='{'flag' if r.flag else ''}'><td>{r.title}</td><td>{pct(r.p_model)}</td><td>{cents(r.bid)} / {cents(r.ask)}</td>"
    f"<td class='gap {'pos' if gap(r).startswith('+') else 'neg'}'>{gap(r)}</td><td>{r.days_observed or ''}</td>"
    f"<td>{'' if pd.isna(r.best_daily_rank) else '#'+str(int(r.best_daily_rank))}</td><td>{pct(r.p_stage1)}</td></tr>"
    for r in C.itertuples() if (r.p_model >= 0.005 or not pd.isna(r.ask) and r.ask >= 0.03))
fl = [f for f in C.flag if f]
flags_html = "".join(f"<li>{f}</li>" for f in fl) or "<li class='none'>No flags right now. The model and Kalshi agree within the 10-point threshold, or there isn't enough data yet.</li>"
html = f"""<title>Netflix Model vs Kalshi</title><meta name="claude:viewport-fit-sides" content="cover"><style>
:root{{--bg:#fbfaf7;--fg:#1d1c1a;--mut:#6b6862;--line:#e3e0d8;--acc:#2f6fdc;--pos:#1d7a44;--neg:#b3412e;--hl:#fff4d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#161614;--fg:#ecebe6;--mut:#a3a098;--line:#33322e;--acc:#7aa5f0;--pos:#5fc28a;--neg:#e88a76;--hl:#3a3220;color-scheme:dark}}}}
:root[data-theme=dark]{{--bg:#161614;--fg:#ecebe6;--mut:#a3a098;--line:#33322e;--acc:#7aa5f0;--pos:#5fc28a;--neg:#e88a76;--hl:#3a3220;color-scheme:dark}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,Segoe UI,sans-serif}}
main{{max-width:760px;margin:0 auto;padding-block:24px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
td,th{{font-variant-numeric:tabular-nums}} h1{{font-size:22px;margin:0 0 4px}} .sub{{color:var(--mut);font-size:13px}}
h2{{font-size:16px;margin:28px 0 8px}} ul{{padding-left:20px}} li.none{{color:var(--mut)}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--mut);font-weight:600;font-size:12px}}
td:first-child{{white-space:normal}} tr.flag td{{background:var(--hl)}} .gap.pos{{color:var(--pos)}} .gap.neg{{color:var(--neg)}}
.note{{color:var(--mut);font-size:13px}}
</style><main>
<h1>Netflix #1 model vs. Kalshi</h1>
<div class='sub'>Week {week_start:%b %d}–{week_end:%b %d} · model run {run_at} ET · Kalshi prices {last_check:%b %d %I:%M %p} ET · daily lists: {len(days_obs)} day(s)</div>
<h2>Flags</h2><ul>{flags_html}</ul>
<h2>Every contender</h2><div class='wrap'><table><thead><tr><th>Movie</th><th>Model</th><th>Kalshi bid / ask</th><th>Gap (pts)</th><th>Days seen</th><th>Best daily</th><th>Stage 1 only</th></tr></thead><tbody>{trs}</tbody></table></div>
<p class='note'>Gap = model minus the Kalshi midpoint; positive means the model thinks Yes is cheap. A flag needs a gap of at least {EDGE_MIN*100:.0f} points against the price you'd actually pay, at least ${VOL_MIN:.0f} traded, and enough daily data ({DATA_MIN:.0%} of the movie's expected week). Stage 1 only = the regression before any daily data. The day-of-week weights and launch curve are assumptions until more weeks are logged. Research tool, not financial advice.</p>
</main>"""
open(os.path.join(OUT, "report.html"), "w").write(html)
print(C[["title", "new", "days", "days_observed", "w", "p_stage1", "p_model", "bid", "ask", "flag"]].round(3).to_string(index=False))
print(meta)
