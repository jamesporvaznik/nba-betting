'use client';

import { useEffect, useState, useMemo } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { supabase } from '@/lib/supabaseClient';

// ── Types ────────────────────────────────────────────────────────────────────

interface PlayerInfo {
  display_name: string;
  position: string | null;
  height_inches: number | null;
  weight_lbs: number | null;
  birth_date: string | null;
  team_abbr: string | null;
}

// interface PlayerRow {
//   display_name: string;
//   position: string | null;
//   height_inches: number | null;
//   weight_lbs: number | null;
//   birth_date: string | null;
//   teams: { abbreviation: string } | null;
// }

interface RawGameLog {
  gamelogid: number;
  gameid: string;
  date: string;
  opponent: string;
  homeaway: string;
  result: string;
  minutes: number;
  points: number;
  rebounds: number;
  assists: number;
  steals: number;
  blocks: number;
  turnovers: number;
  fgm: number;
  fga: number;
  fg_pct: number;
  '3PM': number;
  '3PA': number;
  '3P_PCT': number;
  ftm: number;
  fta: number;
  ft_pct: number;
  plusminus: number;
  gamescore: number;
}

interface GameLog {
  gamelogid: number;
  gameid: string;
  date: string;
  opponent: string;
  homeaway: string;
  result: string;
  minutes: number;
  points: number;
  rebounds: number;
  assists: number;
  steals: number;
  blocks: number;
  turnovers: number;
  fgm: number;
  fga: number;
  fg_pct: number;
  threepm: number;
  threepa: number;
  three_pct: number;
  ftm: number;
  fta: number;
  ft_pct: number;
  plusminus: number;
  gamescore: number;
}

type StatKey =
  | 'points'
  | 'rebounds'
  | 'assists'
  | 'steals'
  | 'blocks'
  | 'turnovers'
  | 'minutes'
  | 'fg_pct'
  | 'gamescore';

const STAT_OPTIONS: { key: StatKey; label: string }[] = [
  { key: 'points',    label: 'Points'    },
  { key: 'rebounds',  label: 'Rebounds'  },
  { key: 'assists',   label: 'Assists'   },
  { key: 'steals',    label: 'Steals'    },
  { key: 'blocks',    label: 'Blocks'    },
  { key: 'turnovers', label: 'Turnovers' },
  { key: 'minutes',   label: 'Minutes'   },
  { key: 'fg_pct',    label: 'FG%'       },
  { key: 'gamescore', label: 'GmSc'      },
];

const GAME_COUNT_OPTIONS = [5, 10, 15, 20];

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatHeight(inches: number | null): string {
  if (!inches) return '—';
  return `${Math.floor(inches / 12)}'${inches % 12}"`;
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
  });
}

function avg(games: GameLog[], key: StatKey): string {
  if (!games.length) return '—';
  const val =
    games.reduce((sum, g) => sum + (Number(g[key]) || 0), 0) / games.length;
  return key === 'fg_pct' ? `${(val * 100).toFixed(1)}%` : val.toFixed(1);
}

function pct(val: number): string {
  return `${(Number(val) * 100).toFixed(1)}%`;
}

function sumStat(logs: GameLog[], key: keyof GameLog): number {
  return logs.reduce((acc, g) => acc + (Number(g[key]) || 0), 0);
}

// ── Component ────────────────────────────────────────────────────────────────

export default function PlayerDetailPage() {
  const params = useParams();
  const playerId = params?.id as string;

  const [player, setPlayer]     = useState<PlayerInfo | null>(null);
  const [gameLogs, setGameLogs] = useState<GameLog[]>([]);
  const [teams, setTeams]       = useState<string[]>([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);

  const [recentStat, setRecentStat]   = useState<StatKey>('points');
  const [recentCount, setRecentCount] = useState(10);

  const [vsStat, setVsStat]   = useState<StatKey>('points');
  const [vsTeam, setVsTeam]   = useState<string>('');
  const [vsCount, setVsCount] = useState(10);

  // ── Fetch ─────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!playerId) return;

    async function fetchData() {
      setLoading(true);
      setError(null);

      const { data: playerData, error: playerErr } = await supabase
  .from('players')
  .select(`
    display_name,
    position,
    height_inches,
    weight_lbs,
    birth_date,
    team:team_id 
  `)
  .eq('nba_api_id', playerId)
  .single();

if (playerErr || !playerData) {
  setError('Player not found.');
  setLoading(false);
  return;
}

setPlayer({
  display_name:  playerData.display_name,
  position:      playerData.position,
  height_inches: playerData.height_inches,
  weight_lbs:    playerData.weight_lbs,
  birth_date:    playerData.birth_date,
  team_abbr:     playerData.team,
});

      const { data: logs, error: logsErr } = await supabase
        .from('player_gamelog')
        .select('*')
        .eq('playerid', playerId)
        .eq('season', '2025-26')
        .order('date', { ascending: false })
        .returns<RawGameLog[]>();

      if (logsErr) {
        setError(`Failed to load game logs: ${logsErr.message}`);
        setLoading(false);
        return;
      }

      const normalized: GameLog[] = (logs ?? []).map((g) => ({
        gamelogid:  g.gamelogid,
        gameid:     g.gameid,
        date:       g.date,
        opponent:   g.opponent,
        homeaway:   g.homeaway,
        result:     g.result,
        minutes:    g.minutes,
        points:     g.points,
        rebounds:   g.rebounds,
        assists:    g.assists,
        steals:     g.steals,
        blocks:     g.blocks,
        turnovers:  g.turnovers,
        fgm:        g.fgm,
        fga:        g.fga,
        fg_pct:     g.fg_pct,
        threepm:    g['3PM'],
        threepa:    g['3PA'],
        three_pct:  g['3P_PCT'],
        ftm:        g.ftm,
        fta:        g.fta,
        ft_pct:     g.ft_pct,
        plusminus:  g.plusminus,
        gamescore:  g.gamescore,
      }));

      setGameLogs(normalized);

      const uniqueTeams = [
        ...new Set(normalized.map((g) => g.opponent)),
      ].sort();
      setTeams(uniqueTeams);
      if (uniqueTeams.length) setVsTeam(uniqueTeams[0]);

      setLoading(false);
    }

    fetchData();
  }, [playerId]);

  // ── Derived ───────────────────────────────────────────────────────────────

  const recentGames = useMemo(
    () => gameLogs.slice(0, recentCount),
    [gameLogs, recentCount]
  );

  const vsGames = useMemo(() => {
    if (!vsTeam) return [];
    return gameLogs
      .filter((g) => g.opponent === vsTeam)
      .slice(0, vsCount);
  }, [gameLogs, vsTeam, vsCount]);

  const seasonAvgs = useMemo(() => {
    if (!gameLogs.length) return null;
    const n = gameLogs.length;
    return {
      gp:        n,
      pts:       (sumStat(gameLogs, 'points')    / n).toFixed(1),
      reb:       (sumStat(gameLogs, 'rebounds')  / n).toFixed(1),
      ast:       (sumStat(gameLogs, 'assists')   / n).toFixed(1),
      stl:       (sumStat(gameLogs, 'steals')    / n).toFixed(1),
      blk:       (sumStat(gameLogs, 'blocks')    / n).toFixed(1),
      to:        (sumStat(gameLogs, 'turnovers') / n).toFixed(1),
      min:       (sumStat(gameLogs, 'minutes')   / n).toFixed(1),
      fg_pct:    pct(sumStat(gameLogs, 'fg_pct')    / n),
      three_pct: pct(sumStat(gameLogs, 'three_pct') / n),
      ft_pct:    pct(sumStat(gameLogs, 'ft_pct')    / n),
    };
  }, [gameLogs]);

  // ── Loading / error states ────────────────────────────────────────────────

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', background: '#080a0e', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <style>{spinnerCss}</style>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
          <div className="spinner" />
          <span className="loading-label">Loading player</span>
        </div>
      </div>
    );
  }

  if (error || !player) {
    return (
      <div style={{ minHeight: '100vh', background: '#080a0e', padding: '2rem', color: '#e8eaf0' }}>
        <Link href="/players" style={{ color: '#f05a1a', fontFamily: 'Barlow Condensed, sans-serif', fontSize: '0.85rem', textDecoration: 'none' }}>
          ← Back to players
        </Link>
        <p style={{ marginTop: '2rem', color: '#9a5040' }}>{error ?? 'Player not found.'}</p>
      </div>
    );
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <>
      <style>{css}</style>
      <main className="root">
        <div className="inner">

          <Link href="/players" className="back-link">
            <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            All Players
          </Link>

          {/* Hero */}
          <div className="hero">
            <div className="hero-badge">{player.team_abbr ?? 'NBA'}</div>
            <div className="hero-info">
              <div className="hero-name">{player.display_name}</div>
              <div className="hero-meta">
                {[
                  player.position,
                  formatHeight(player.height_inches),
                  player.weight_lbs ? `${player.weight_lbs} lbs` : null,
                  player.birth_date
                    ? new Date(player.birth_date).toLocaleDateString('en-US', {
                        month: 'short', day: 'numeric', year: 'numeric',
                      })
                    : null,
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </div>
            </div>
          </div>

          {/* Season averages */}
          {seasonAvgs && (
            <section className="section">
              <div className="section-header">
                <div>
                  <div className="section-label">Season Averages</div>
                  <div className="section-sub">2025–26 · {seasonAvgs.gp} GP</div>
                </div>
              </div>
              <div className="avg-grid">
                {(
                  [
                    { label: 'PTS', val: seasonAvgs.pts       },
                    { label: 'REB', val: seasonAvgs.reb       },
                    { label: 'AST', val: seasonAvgs.ast       },
                    { label: 'STL', val: seasonAvgs.stl       },
                    { label: 'BLK', val: seasonAvgs.blk       },
                    { label: 'TO',  val: seasonAvgs.to        },
                    { label: 'MIN', val: seasonAvgs.min       },
                    { label: 'FG%', val: seasonAvgs.fg_pct    },
                    { label: '3P%', val: seasonAvgs.three_pct },
                    { label: 'FT%', val: seasonAvgs.ft_pct    },
                  ] as { label: string; val: string }[]
                ).map(({ label, val }) => (
                  <div key={label} className="avg-card">
                    <div className="avg-val">{val}</div>
                    <div className="avg-label">{label}</div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Recent games */}
          <section className="section">
            <div className="section-header">
              <div className="section-label">Recent Games</div>
              <div className="filter-row">
                <div className="filter-group">
                  <span className="filter-tag">Stat</span>
                  <select
                    className="sel"
                    value={recentStat}
                    onChange={(e) => setRecentStat(e.target.value as StatKey)}
                  >
                    {STAT_OPTIONS.map(({ key, label }) => (
                      <option key={key} value={key}>{label}</option>
                    ))}
                  </select>
                </div>
                <div className="filter-group">
                  <span className="filter-tag">Last</span>
                  <div className="pill-group">
                    {GAME_COUNT_OPTIONS.map((n) => (
                      <button
                        key={n}
                        className={`pill${recentCount === n ? ' active' : ''}`}
                        onClick={() => setRecentCount(n)}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {recentGames.length === 0 ? (
              <div className="empty">No games found.</div>
            ) : (
              <>
                <div className="avg-banner">
                  Avg{' '}
                  <strong>{avg(recentGames, recentStat)}</strong>{' '}
                  {STAT_OPTIONS.find((s) => s.key === recentStat)?.label} over last {recentGames.length} games
                </div>
                <GameTable games={recentGames} highlightStat={recentStat} />
              </>
            )}
          </section>

          {/* Vs team */}
          <section className="section">
            <div className="section-header">
              <div className="section-label">vs. Opponent</div>
              <div className="filter-row">
                <div className="filter-group">
                  <span className="filter-tag">Team</span>
                  <select
                    className="sel"
                    value={vsTeam}
                    onChange={(e) => setVsTeam(e.target.value)}
                  >
                    {teams.map((t) => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>
                <div className="filter-group">
                  <span className="filter-tag">Stat</span>
                  <select
                    className="sel"
                    value={vsStat}
                    onChange={(e) => setVsStat(e.target.value as StatKey)}
                  >
                    {STAT_OPTIONS.map(({ key, label }) => (
                      <option key={key} value={key}>{label}</option>
                    ))}
                  </select>
                </div>
                <div className="filter-group">
                  <span className="filter-tag">Last</span>
                  <div className="pill-group">
                    {GAME_COUNT_OPTIONS.map((n) => (
                      <button
                        key={n}
                        className={`pill${vsCount === n ? ' active' : ''}`}
                        onClick={() => setVsCount(n)}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {vsGames.length === 0 ? (
              <div className="empty">No games vs {vsTeam || '—'} found.</div>
            ) : (
              <>
                <div className="avg-banner">
                  Avg{' '}
                  <strong>{avg(vsGames, vsStat)}</strong>{' '}
                  {STAT_OPTIONS.find((s) => s.key === vsStat)?.label} vs {vsTeam} — last {vsGames.length} games
                </div>
                <GameTable games={vsGames} highlightStat={vsStat} />
              </>
            )}
          </section>

        </div>
      </main>
    </>
  );
}

// ── GameTable ─────────────────────────────────────────────────────────────────

function GameTable({
  games,
  highlightStat,
}: {
  games: GameLog[];
  highlightStat: StatKey;
}) {
  const cols: { key: StatKey; label: string }[] = [
    { key: 'points',    label: 'PTS'  },
    { key: 'rebounds',  label: 'REB'  },
    { key: 'assists',   label: 'AST'  },
    { key: 'steals',    label: 'STL'  },
    { key: 'blocks',    label: 'BLK'  },
    { key: 'turnovers', label: 'TO'   },
    { key: 'minutes',   label: 'MIN'  },
    { key: 'fg_pct',    label: 'FG%'  },
    { key: 'gamescore', label: 'GmSc' },
  ];

  return (
    <div className="table-wrap">
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th className="t-left">Date</th>
              <th className="t-left">Opp</th>
              <th className="t-center">H/A</th>
              <th className="t-center">W/L</th>
              {cols.map(({ key, label }) => (
                <th
                  key={key}
                  className={`t-right${key === highlightStat ? ' th-active' : ''}`}
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {games.map((g) => (
              <tr key={g.gamelogid}>
                <td className="t-left td-date">{formatDate(g.date)}</td>
                <td className="t-left td-opp">{g.opponent}</td>
                <td className="t-center td-meta">
                  {g.homeaway === 'H' ? 'Home' : 'Away'}
                </td>
                <td className={`t-center td-result ${g.result === 'W' ? 'win' : 'loss'}`}>
                  {g.result}
                </td>
                {cols.map(({ key }) => {
                  const raw = g[key];
                  const display =
                    key === 'fg_pct'
                      ? pct(raw as number)
                      : Number(raw).toFixed(key === 'gamescore' ? 1 : 0);
                  return (
                    <td
                      key={key}
                      className={`t-right td-num${key === highlightStat ? ' td-highlight' : ''}`}
                    >
                      {display}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const spinnerCss = `
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@400;500&display=swap');
  .spinner { width:2rem;height:2rem;border:2px solid #1e2230;border-top-color:#f05a1a;border-radius:50%;animation:spin .7s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .loading-label { font-family:'Barlow Condensed',sans-serif;font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:#3a3f58; }
`;

const css = `
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@400;500&display=swap');

  .root { min-height:100vh; background:#080a0e; color:#e8eaf0; font-family:'Barlow',sans-serif; padding:2rem 1.5rem 5rem; }
  .inner { max-width:1100px; margin:0 auto; }

  .back-link { display:inline-flex; align-items:center; gap:.4rem; text-decoration:none; font-family:'Barlow Condensed',sans-serif; font-size:.78rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#3a3f58; margin-bottom:2rem; transition:color .15s; }
  .back-link:hover { color:#f05a1a; }

  .hero { display:flex; align-items:center; gap:1.25rem; margin-bottom:2.5rem; padding-bottom:2rem; border-bottom:1px solid #111318; }
  .hero-badge { font-family:'Barlow Condensed',sans-serif; font-size:1rem; font-weight:700; letter-spacing:.08em; background:#111318; border:1px solid #1e2230; color:#f05a1a; padding:.3rem .7rem; border-radius:5px; flex-shrink:0; }
  .hero-name { font-family:'Barlow Condensed',sans-serif; font-size:2rem; font-weight:700; color:#f0f2f7; line-height:1; letter-spacing:-.01em; }
  .hero-meta { font-size:.82rem; color:#4e5470; margin-top:.3rem; }

  .avg-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(80px,1fr)); gap:.5rem; }
  .avg-card { background:#0d0f15; border:1px solid #1a1d28; border-radius:8px; padding:.75rem .5rem; text-align:center; }
  .avg-val { font-family:'Barlow Condensed',sans-serif; font-size:1.35rem; font-weight:700; color:#e8eaf0; line-height:1; }
  .avg-label { font-family:'Barlow Condensed',sans-serif; font-size:.65rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#3a3f58; margin-top:.3rem; }

  .section { margin-bottom:3rem; }
  .section-header { display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:1rem; margin-bottom:1rem; padding-bottom:.75rem; border-bottom:1px solid #111318; }
  .section-label { font-family:'Barlow Condensed',sans-serif; font-size:1.1rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; color:#c8cad8; }
  .section-sub { font-family:'Barlow Condensed',sans-serif; font-size:.75rem; color:#3a3f58; letter-spacing:.06em; text-transform:uppercase; margin-top:.2rem; }

  .filter-row { display:flex; flex-wrap:wrap; align-items:center; gap:.75rem; }
  .filter-group { display:flex; align-items:center; gap:.4rem; }
  .filter-tag { font-family:'Barlow Condensed',sans-serif; font-size:.68rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#3a3f58; }

  .sel { background:#0d0f15; border:1px solid #1e2230; border-radius:6px; color:#c8cad8; font-family:'Barlow Condensed',sans-serif; font-size:.85rem; padding:.3rem .55rem; outline:none; cursor:pointer; transition:border-color .15s; }
  .sel:focus { border-color:#f05a1a; }

  .pill-group { display:flex; gap:.25rem; }
  .pill { background:#0d0f15; border:1px solid #1e2230; border-radius:5px; color:#4e5470; font-family:'Barlow Condensed',sans-serif; font-size:.8rem; font-weight:600; padding:.25rem .55rem; cursor:pointer; transition:background .12s,border-color .12s,color .12s; }
  .pill:hover { border-color:#3a3f58; color:#8a90a8; }
  .pill.active { background:#1a1008; border-color:#f05a1a; color:#f05a1a; }

  .avg-banner { font-size:.82rem; color:#4e5470; margin-bottom:.75rem; font-family:'Barlow Condensed',sans-serif; letter-spacing:.02em; }
  .avg-banner strong { color:#f05a1a; font-weight:700; }

  .table-wrap { border:1px solid #1a1d28; border-radius:10px; overflow:hidden; }
  .table-scroll { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; min-width:660px; }

  thead tr { background:#0d0f15; }
  th { padding:.6rem 1rem; font-family:'Barlow Condensed',sans-serif; font-size:.68rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#3a3f58; border-bottom:1px solid #1a1d28; white-space:nowrap; }
  th.th-active { color:#f05a1a; }
  th.t-left { text-align:left; }
  th.t-right { text-align:right; }
  th.t-center { text-align:center; }

  tbody tr { border-bottom:1px solid #0f1118; transition:background .1s; }
  tbody tr:last-child { border-bottom:none; }
  tbody tr:hover { background:#0d0f16; }

  td { padding:.65rem 1rem; white-space:nowrap; }
  td.t-left { text-align:left; }
  td.t-right { text-align:right; }
  td.t-center { text-align:center; }

  .td-date { font-family:'Barlow Condensed',sans-serif; font-size:.82rem; color:#4e5470; }
  .td-opp { font-family:'Barlow Condensed',sans-serif; font-size:.88rem; font-weight:600; color:#8a90a8; }
  .td-meta { font-family:'Barlow Condensed',sans-serif; font-size:.75rem; color:#3a3f58; }
  .td-result { font-family:'Barlow Condensed',sans-serif; font-size:.8rem; font-weight:700; letter-spacing:.05em; }
  .td-result.win { color:#3a9e6a; }
  .td-result.loss { color:#9e3a3a; }
  .td-num { font-family:'Barlow Condensed',sans-serif; font-size:.88rem; color:#6b7090; }
  .td-highlight { color:#f0f2f7 !important; font-weight:600; }

  .empty { color:#3a3f58; font-family:'Barlow Condensed',sans-serif; font-size:.9rem; padding:2rem 0; text-align:center; letter-spacing:.04em; }

  .spinner { width:2rem;height:2rem;border:2px solid #1e2230;border-top-color:#f05a1a;border-radius:50%;animation:spin .7s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .loading-label { font-family:'Barlow Condensed',sans-serif;font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:#3a3f58; }

  @media (max-width:600px) {
    .hero { flex-direction:column; align-items:flex-start; }
    .section-header { flex-direction:column; }
    .avg-grid { grid-template-columns:repeat(5,1fr); }
  }
`;