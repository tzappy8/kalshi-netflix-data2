"""
Netflix US #1 movie model + paper-trading scorecard.   python3 model/run.py   (from the repo root)

Reads:  data/kalshi_prices*.csv, data/kalshi_log.csv, data/top10_log.csv, data/weekly_log.csv  (laptop logger)
        model/ history, Stage 1 coefficients, title facts, seed daily lists, week_config.json
Writes: model_output/report.html, model_output/latest.json, model_output/paper_trades.csv

Model
 1. Stage 1 prior: a conditional logit on 5 years of weekly charts gives each movie a score V1.
 2. Each daily Top 10 rank becomes estimated views (rank curve measured from Netflix's weekly view
    counts), adjusted for day of week and a launch/decay curve, giving each movie a peak-day level.
 3. Prior and daily data are combined (more days seen = more weight on data). Weekly views =
    level x the movie's own days in the window. Monte Carlo over the uncertainty gives P(#1).
 4. Flags: model vs. the price you would actually pay, only when the gap clears 10 points, enough
    has traded, and enough daily data exists.
Paper trading
 The model is replayed "as of" each scheduled check time using only data logged by then. A flag
 becomes a $10 paper buy at the real ask (capped by contracts actually offered), once per movie and
 side per week. Trades settle on the official weekly chart. Kalshi's fee is approximated as
 0.07 x contracts x price x (1 - price), rounded up to the cent.
Assumptions to calibrate as weeks are logged: day-of-week weights, launch/decay curve, noise sizes.
"""
import glob, json, math, os, sys, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "model"); DATA = os.path.join(ROOT, "data"); OUT = os.path.join(ROOT, "model_output")
ET = ZoneInfo("America/New_York")
RANK_SHARE = {1: 1.0, 2: .579, 3: .371, 4: .287, 5: .238, 6: .21, 7: .183, 8: .16, 9: .146, 10: .131}
OFF_LIST = 0.10
DOW = [0.92, 0.88, 0.88, 0.92, 1.08, 1.20, 1.12]          # Mon..Sun (assumption)
FIELD_P = 0.03
EDGE_MIN, VOL_MIN, DATA_MIN = 0.10, 200.0, 0.30
STAKE = 10.0                                               # dollars per paper trade
CHECK_TIMES = [(7, 41), (13, 41), (18, 41), (23, 41)]      # same as the scheduled check
PRIOR_SD, DATA_SD = 0.9, 0.25

def read_csv(p, **k):
    return pd.read_csv(p, **k) if os.path.exists(p) else None
norm = lambda x: str(x).replace("Tyler Perry's ", "").replace("’", "'").strip().lower()

def age_curve(age, kind):
    if kind == "old": return 0.97 ** age
    if age < 0: return 0.0
    if age == 0: return 0.85
    if age <= 2: return 1.0
    a = age - 2
    return 0.88 ** min(a, 12) * 0.92 ** max(0, min(a - 12, 7)) * 0.97 ** max(0, a - 19)

def event_week(event):
    close_day = dt.datetime.strptime(event.split("-")[1], "%y%b%d").date()   # 26NOV16 = Mon Nov 16
    week_end = close_day - dt.timedelta(days=1)
    return week_end - dt.timedelta(days=6), week_end, close_day

# ---------------- load everything once ----------------
def load():
    ks = []
    for p in [os.path.join(DATA, "kalshi_log.csv")] + sorted(glob.glob(os.path.join(DATA, "kalshi_prices*.csv"))):
        x = read_csv(p)
        if x is not None and not x.empty: ks.append(x)
    if not ks: sys.exit("no Kalshi data yet")
    K = pd.concat(ks, ignore_index=True)
    K["checked_at"] = pd.to_datetime(K.checked_at)
    K = K[K.event.astype(str).str.startswith("KXNETFLIXRANKMOVIE-") & K.title.notna() & (K.title != "DEACTIVATED")]
    for c in ("yes_bid", "yes_ask", "volume", "yes_bid_size", "yes_ask_size"):
        K[c] = pd.to_numeric(K[c], errors="coerce") if c in K else np.nan
    K[["yes_bid", "yes_ask", "volume"]] = K[["yes_bid", "yes_ask", "volume"]].fillna(0)
    K["has_size"] = K.yes_ask_size.notna()

    hist = pd.read_csv(os.path.join(M, "us_weekly_history.csv"), parse_dates=["week"])
    wl = read_csv(os.path.join(DATA, "weekly_log.csv"))
    if wl is not None and not wl.empty:
        wl = wl.rename(columns={"week_end": "week", "rank": "weekly_rank", "title": "show_title", "weeks": "cumulative_weeks_in_top_10"})
        wl["week"] = pd.to_datetime(wl.week)
        hist = pd.concat([hist, wl[hist.columns]]).drop_duplicates(["week", "weekly_rank"], keep="last")

    frames = []
    s = read_csv(os.path.join(M, "seed_daily.csv"))
    if s is not None:
        s["view_day"] = pd.to_datetime(s.list_date) - pd.Timedelta(days=1)
        s["available_at"] = pd.to_datetime(s.list_date) + pd.Timedelta(hours=7)   # assume seen by 7 AM that day
        frames.append(s[["view_day", "rank", "title", "available_at"]])
    t = read_csv(os.path.join(DATA, "top10_log.csv"))
    if t is not None and not t.empty:
        t["available_at"] = pd.to_datetime(t.seen_at)
        t["view_day"] = (t.available_at - pd.Timedelta(hours=12)).dt.normalize()
        frames.append(t[["view_day", "rank", "title", "available_at"]])
    daily = pd.concat(frames) if frames else pd.DataFrame(columns=["view_day", "rank", "title", "available_at"])
    daily["view_day"] = pd.to_datetime(daily.view_day).dt.date
    daily = daily.sort_values("available_at")
    return dict(K=K, hist=hist, daily=daily,
                cfg=json.load(open(os.path.join(M, "week_config.json"))).get("titles", {}),
                flags=pd.read_csv(os.path.join(M, "title_flags.csv")).set_index("title"),
                co=json.load(open(os.path.join(M, "stage1_coefs.json"))))

# ---------------- the model, as of a moment in time ----------------
def model_at(D, event, asof, n_sim=20000, seed=7):
    rng = np.random.default_rng(seed)
    week_start, week_end, _ = event_week(event)
    kk = D["K"][(D["K"].event == event) & (D["K"].checked_at <= asof)]
    if kk.empty: return None, None
    last_check = kk.checked_at.max()
    px = kk[kk.checked_at >= last_check - pd.Timedelta(minutes=5)].sort_values(["checked_at", "has_size"]).drop_duplicates("title", keep="last")

    hist = D["hist"]; prev_end = pd.Timestamp(week_start - dt.timedelta(days=1))
    last = hist[hist.week == prev_end]
    if last.empty: last = hist[hist.week == hist[hist.week < pd.Timestamp(week_start)].week.max()]

    daily = D["daily"][D["daily"].available_at <= asof]
    daily = daily.drop_duplicates(["view_day", "rank"], keep="last")
    daily = daily[(daily.view_day >= week_start) & (daily.view_day <= week_end)]
    days_obs = sorted(daily.view_day.unique())

    cfg, flags, co = D["cfg"], D["flags"], D["co"]
    names = {}
    for n in list(last.show_title) + list(px.title) + list(daily.title):
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
        if new and rel < week_start: rel = week_start
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
        age0 = (week_start - rel).days if new else 7 * weeks_before + 3
        rows.append(dict(key=key, title=name, new=new, release=rel, days=days, kind=kind, age0=age0, V1=v1))
    C = pd.DataFrame(rows)

    window = [week_start + dt.timedelta(days=i) for i in range(7)]
    lev, wf, nob, best, ww = [], [], [], [], []
    for r in C.itertuples():
        td = []
        for d in window:
            if r.new and d < r.release: continue
            age = (d - r.release).days if r.new else r.age0 + (d - week_start).days
            td.append((d, DOW[d.weekday()] * age_curve(age, r.kind)))
        exp_all = sum(w for _, w in td) or 1e-9
        obs = [(d, w) for d, w in td if d in days_obs]
        lv = []
        for d, w in obs:
            hit = daily[(daily.view_day == d) & (daily.title.map(norm) == r.key)]
            v = (RANK_SHARE[int(hit["rank"].iloc[0])] if len(hit) else OFF_LIST) * DOW[d.weekday()]
            lv.append(v / max(w, 1e-9))
        ranks = daily[daily.title.map(norm) == r.key]["rank"]
        best.append(int(ranks.min()) if len(ranks) else None)
        lev.append(float(np.exp(np.average(np.log(lv), weights=np.arange(1, len(lv) + 1)))) if lv else np.nan)
        wf.append(sum(w for _, w in obs) / exp_all if lv else 0.0); nob.append(len(obs)); ww.append(exp_all)
    C["level"], C["w"], C["days_observed"], C["best_daily_rank"], C["window_weight"] = lev, wf, nob, best, ww

    c0 = -C.V1.max()
    pm_, ps_ = [], []
    for r in C.itertuples():
        mu0 = c0 + r.V1
        if np.isnan(r.level):
            pm_.append(mu0); ps_.append(PRIOR_SD)
        else:
            sd_d = DATA_SD / math.sqrt(r.days_observed) + 0.35 * (1 - r.w)
            wt = PRIOR_SD ** -2 + sd_d ** -2
            pm_.append((mu0 / PRIOR_SD ** 2 + math.log(r.level) / sd_d ** 2) / wt); ps_.append(wt ** -0.5)
    logL = np.array(pm_) + rng.normal(size=(n_sim, len(C))) * np.array(ps_)
    win = np.bincount((np.exp(logL) * C.window_weight.values).argmax(1), minlength=len(C)) / n_sim
    C["p_model"] = (1 - FIELD_P) * win
    C["p_stage1"] = (1 - FIELD_P) * np.exp(C.V1 - C.V1.max()) / np.exp(C.V1 - C.V1.max()).sum()

    pmx = px.assign(key=px.title.map(norm)).set_index("key")
    for col, src in (("bid", "yes_bid"), ("ask", "yes_ask"), ("volume", "volume"), ("bid_size", "yes_bid_size"), ("ask_size", "yes_ask_size")):
        C[col] = C.key.map(pmx[src])
    C["mid"] = (C.bid + C.ask) / 2
    C["edge_yes"] = C.p_model - C.ask
    C["edge_no"] = C.bid - C.p_model
    C["flag_side"] = ""
    ok = C.ask.notna() & (C.volume.fillna(0) >= VOL_MIN) & (C.w >= DATA_MIN)
    C.loc[ok & (C.edge_yes >= EDGE_MIN) & (C.edge_yes >= C.edge_no), "flag_side"] = "Yes"
    C.loc[ok & (C.edge_no >= EDGE_MIN) & (C.edge_no > C.edge_yes), "flag_side"] = "No"
    meta = dict(event=event, week_start=week_start, week_end=week_end, prices_at=last_check, days_obs=days_obs)
    return C.sort_values("p_model", ascending=False).reset_index(drop=True), meta

def flag_text(r):
    side = r.flag_side
    price = r.ask if side == "Yes" else 1 - r.bid
    edge = r.edge_yes if side == "Yes" else r.edge_no
    p = r.p_model if side == "Yes" else 1 - r.p_model
    size = r.ask_size if side == "Yes" else r.bid_size
    thin = "" if pd.isna(size) else (f" Thin: only {size:.0f} contract{'s' if size != 1 else ''} at that price." if size < 20 else f" {size:.0f} contracts at that price.")
    why = f"{r.days_observed} day(s) of daily data, best daily rank #{int(r.best_daily_rank)}" if r.days_observed else "no daily data"
    return f"Buy {side} on {r.title}: model {p:.0%} vs {price*100:.0f}¢ to buy ({edge*100:.0f} pts). {why}.{thin}"

def fee(contracts, price):
    return math.ceil(0.07 * contracts * price * (1 - price) * 100) / 100

# ---------------- paper trading replay ----------------
def replay(D, now):
    trades, scores = [], []
    for event in sorted(D["K"].event.unique(), key=lambda e: event_week(e)[2]):
        week_start, week_end, close_day = event_week(event)
        ev = D["K"][D["K"].event == event]
        first = ev.checked_at.min()
        close_at = dt.datetime.combine(close_day, dt.time(23, 59))
        times = []
        d = first.date()
        while d <= close_day:
            for h, m in CHECK_TIMES:
                t = dt.datetime.combine(d, dt.time(h, m))
                if first <= t <= min(now, close_at): times.append(t)
            d += dt.timedelta(days=1)
        held = set()
        chart = D["hist"][(D["hist"].week == pd.Timestamp(week_end)) & (D["hist"].weekly_rank == 1)]
        winner = norm(chart.show_title.iloc[0]) if not chart.empty else None
        for t in times:
            C, meta = model_at(D, event, pd.Timestamp(t), n_sim=4000)
            if C is None: continue
            if winner is not None:   # accuracy at this check time: model vs market midpoint (normalized)
                mk = C[C.mid.notna()]
                if len(mk) and (mk.key == winner).any():
                    pm = float(C.loc[C.key == winner, "p_model"].iloc[0]) if (C.key == winner).any() else 0.005
                    mids = mk.set_index("key").mid.clip(lower=0.005); mids = mids / mids.sum()
                    scores.append(dict(event=event, at=t, model_ll=-math.log(max(pm, 0.005)), market_ll=-math.log(float(mids.get(winner, 0.005)))))
            for r in C[C.flag_side != ""].itertuples():
                k = (r.key, r.flag_side)
                if k in held: continue
                price = float(r.ask) if r.flag_side == "Yes" else float(1 - r.bid)
                if price <= 0 or price >= 1: continue
                avail = r.ask_size if r.flag_side == "Yes" else r.bid_size
                n = math.floor(STAKE / price)
                if not pd.isna(avail): n = min(n, int(avail))
                if n <= 0: continue
                held.add(k)
                cost = n * price + fee(n, price)
                res, pnl = "open", None
                if winner is not None:
                    won = (r.key == winner) if r.flag_side == "Yes" else (r.key != winner)
                    res = "won" if won else "lost"; pnl = (n * 1.0 if won else 0.0) - cost
                trades.append(dict(event=event, week=f"{week_start:%b %d}–{week_end:%b %d}", at=t, title=r.title, side=r.flag_side,
                                   price=price, contracts=n, cost=round(cost, 2), model_p=r.p_model if r.flag_side == "Yes" else 1 - r.p_model,
                                   result=res, pnl=None if pnl is None else round(pnl, 2)))
    return pd.DataFrame(trades), pd.DataFrame(scores)

# ---------------- report ----------------
def render(C, meta, T, S, run_at):
    pct = lambda x: "" if pd.isna(x) else f"{x:.0%}"
    cents = lambda x: "" if pd.isna(x) else f"{x*100:.0f}¢"
    def gap(r):
        return "" if pd.isna(r.mid) else f"{(r.p_model - r.mid) * 100:+.0f}"
    rows = "".join(
        f"<tr class='{'flag' if r.flag_side else ''}'><td>{r.title}</td><td>{pct(r.p_model)}</td><td>{cents(r.bid)} / {cents(r.ask)}</td>"
        f"<td class='gap {'pos' if gap(r).startswith('+') else 'neg'}'>{gap(r)}</td><td>{r.days_observed or ''}</td>"
        f"<td>{'' if pd.isna(r.best_daily_rank) else '#' + str(int(r.best_daily_rank))}</td><td>{pct(r.p_stage1)}</td></tr>"
        for r in C.itertuples() if (r.p_model >= 0.005 or (not pd.isna(r.ask) and r.ask >= 0.03)))
    fl = [flag_text(r) for r in C.itertuples() if r.flag_side]
    flags_html = "".join(f"<li>{f}</li>" for f in fl) or "<li class='none'>No flags right now. The model and Kalshi agree within 10 points, or there isn't enough daily data yet.</li>"

    if T.empty:
        paper = "<p class='note'>No paper trades yet. A trade is logged the first time a flag appears for a movie and side in a week.</p>"
        totals = ""
    else:
        settled = T[T.result != "open"]
        n_open = int((T.result == "open").sum())
        pnl = settled.pnl.sum() if len(settled) else 0.0
        staked = settled.cost.sum() if len(settled) else 0.0
        wins = int((settled.result == "won").sum())
        totals = (f"<div class='kpis'><div><b>{len(T)}</b><span>paper trades</span></div><div><b>{wins}/{len(settled)}</b><span>won (settled)</span></div>"
                  f"<div><b class='{'pos' if pnl >= 0 else 'neg'}'>{'+' if pnl >= 0 else '−'}${abs(pnl):.2f}</b><span>profit after fees</span></div>"
                  f"<div><b>{(pnl / staked * 100 if staked else 0):+.0f}%</b><span>return on settled</span></div><div><b>{n_open}</b><span>still open</span></div></div>")
        trs = "".join(
            f"<tr><td>{r.week}</td><td>{r.at:%a %I:%M %p}</td><td>{r.title}</td><td>{r.side}</td><td>{r.price*100:.0f}¢ × {r.contracts}</td>"
            f"<td>${r.cost:.2f}</td><td>{r.model_p:.0%}</td><td>{r.result}</td>"
            f"<td class='{'' if pd.isna(r.pnl) else ('pos' if r.pnl >= 0 else 'neg')}'>{'' if pd.isna(r.pnl) else ('+' if r.pnl >= 0 else '−') + f'${abs(r.pnl):.2f}'}</td></tr>"
            for r in T.sort_values("at", ascending=False).itertuples())
        paper = (f"<div class='wrap'><table><thead><tr><th>Week</th><th>Logged</th><th>Movie</th><th>Side</th><th>Price × qty</th><th>Cost</th>"
                 f"<th>Model</th><th>Result</th><th>P/L</th></tr></thead><tbody>{trs}</tbody></table></div>")
    if S.empty:
        acc = "<p class='note'>Model vs. market accuracy appears once a week has settled.</p>"
    else:
        g = S.groupby("event").agg(checks=("at", "size"), model=("model_ll", "mean"), market=("market_ll", "mean")).reset_index()
        acc = "<div class='wrap'><table><thead><tr><th>Week (market)</th><th>Checks</th><th>Model log loss</th><th>Kalshi log loss</th><th>Better</th></tr></thead><tbody>" + "".join(
            f"<tr><td>{r.event.split('-')[1]}</td><td>{r.checks}</td><td>{r.model:.2f}</td><td>{r.market:.2f}</td><td>{'Model' if r.model < r.market else 'Kalshi'}</td></tr>"
            for r in g.itertuples()) + f"</tbody></table></div><p class='note'>Log loss = how surprised each was by the actual winner, averaged over every check that week. Lower is better. Overall: model {S.model_ll.mean():.2f}, Kalshi {S.market_ll.mean():.2f}.</p>"

    ws, we = meta["week_start"], meta["week_end"]
    return f"""<title>Netflix Model vs Kalshi</title><meta name="claude:viewport-fit-sides" content="cover"><style>
:root{{--bg:#fbfaf7;--fg:#1d1c1a;--mut:#6b6862;--line:#e3e0d8;--acc:#2f6fdc;--pos:#1d7a44;--neg:#b3412e;--hl:#fff4d6;--card:#f2efe8}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#161614;--fg:#ecebe6;--mut:#a3a098;--line:#33322e;--acc:#7aa5f0;--pos:#5fc28a;--neg:#e88a76;--hl:#3a3220;--card:#201f1c;color-scheme:dark}}}}
:root[data-theme=dark]{{--bg:#161614;--fg:#ecebe6;--mut:#a3a098;--line:#33322e;--acc:#7aa5f0;--pos:#5fc28a;--neg:#e88a76;--hl:#3a3220;--card:#201f1c;color-scheme:dark}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,Segoe UI,sans-serif}}
main{{max-width:820px;margin:0 auto;padding-block:24px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
h1{{font-size:22px;margin:0 0 4px;text-wrap:balance}} .sub{{color:var(--mut);font-size:13px}} h2{{font-size:16px;margin:30px 0 8px}}
ul{{padding-left:20px}} li.none{{color:var(--mut)}} .wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line);white-space:nowrap;font-variant-numeric:tabular-nums}}
th{{color:var(--mut);font-weight:600;font-size:12px}} td:first-child{{white-space:normal}} tr.flag td{{background:var(--hl)}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}} .note{{color:var(--mut);font-size:13px;max-width:65ch}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:8px 0 12px}}
.kpis div{{background:var(--card);border-radius:8px;padding:10px 12px;display:flex;flex-direction:column;gap:2px}}
.kpis b{{font-size:20px;font-variant-numeric:tabular-nums}} .kpis span{{color:var(--mut);font-size:12px}}
</style><main>
<h1>Netflix #1 model vs. Kalshi</h1>
<div class='sub'>Week {ws:%b %d}–{we:%b %d} · updated {run_at} ET · Kalshi prices {meta['prices_at']:%b %d %I:%M %p} ET · daily lists: {len(meta['days_obs'])} day(s)</div>
<h2>Flags right now</h2><ul>{flags_html}</ul>
<h2>Every contender</h2><div class='wrap'><table><thead><tr><th>Movie</th><th>Model</th><th>Kalshi bid / ask</th><th>Gap (pts)</th><th>Days seen</th><th>Best daily</th><th>Stage 1 only</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Paper trading</h2>{totals}{paper}
<p class='note'>Each flag becomes a ${STAKE:.0f} pretend buy at the real price you'd pay, capped by the contracts actually for sale, once per movie and side per week. Trades settle on Netflix's official weekly chart. Fees are Kalshi's standard formula, approximated. Results replay the model as of each scheduled check (7:41 AM, 1:41 PM, 6:41 PM, 11:41 PM), using only data logged by then.</p>
<h2>Model vs. market accuracy</h2>{acc}
<p class='note'>Gap = model minus the Kalshi midpoint; positive means the model thinks Yes is cheap. A flag needs a gap of at least {EDGE_MIN*100:.0f} points against the price you'd actually pay, at least ${VOL_MIN:.0f} traded, and enough daily data ({DATA_MIN:.0%} of the movie's expected week). The day-of-week weights and launch curve are assumptions until more weeks are logged. A handful of weeks is mostly luck; judge the scorecard after 8–12 weeks. Research tool, not financial advice.</p>
</main>"""

def main():
    D = load()
    now = pd.Timestamp(dt.datetime.now(ET).replace(tzinfo=None))
    event = sorted(D["K"].event.unique(), key=lambda e: event_week(e)[2])[-1]
    C, meta = model_at(D, event, now)
    T, S = replay(D, now)
    os.makedirs(OUT, exist_ok=True)
    run_at = dt.datetime.now(ET).strftime("%b %d %I:%M %p")
    open(os.path.join(OUT, "report.html"), "w").write(render(C, meta, T, S, run_at))
    T.to_csv(os.path.join(OUT, "paper_trades.csv"), index=False)
    keep = ["title", "new", "release", "days", "days_observed", "best_daily_rank", "w", "p_stage1", "p_model", "bid", "ask", "bid_size", "ask_size", "volume", "flag_side"]
    json.dump(dict(meta={k: str(v) for k, v in meta.items()}, rows=json.loads(C[keep].to_json(orient="records", date_format="iso")),
                   flags=[flag_text(r) for r in C.itertuples() if r.flag_side],
                   paper=dict(trades=len(T), settled_pnl=float(T.pnl.dropna().sum()) if len(T) else 0.0)),
              open(os.path.join(OUT, "latest.json"), "w"), indent=1, default=str)
    print(C[["title", "days_observed", "w", "p_model", "bid", "ask", "flag_side"]].head(6).round(3).to_string(index=False))
    print("paper trades:", len(T)); print(T.to_string(index=False) if len(T) else "")

if __name__ == "__main__":
    main()
