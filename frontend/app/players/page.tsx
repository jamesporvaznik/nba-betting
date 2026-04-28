'use client';

import { useEffect, useState, useMemo } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabaseClient';

interface PlayerStats {
  playerid: number;
  display_name: string;
  games_played: number;
  avg_points: number;
  avg_rebounds: number;
  avg_assists: number;
}

interface AggregatedStat {
  playerid: number;
  games_played: number;
  total_points: number;
  total_rebounds: number;
  total_assists: number;
}

type SortKey = 'avg_points' | 'avg_rebounds' | 'avg_assists' | 'games_played';

export default function PlayersPage() {
  const router = useRouter();
  const [allPlayers, setAllPlayers] = useState<PlayerStats[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<SortKey>('avg_points');
  const [sortDesc, setSortDesc] = useState(true);

  useEffect(() => {
    async function fetchAllPlayers() {
      setLoading(true);
      setErrorMsg(null);

      const { data: stats, error: statsError } = await supabase
        .rpc('get_player_stats_for_season', { season_param: '2025-26' });

      if (statsError) {
        console.error(statsError);
        setErrorMsg(`Database error: ${statsError.message}`);
        setLoading(false);
        return;
      }

      if (!stats || stats.length === 0) {
        setErrorMsg('No game log data found.');
        setLoading(false);
        return;
      }

      const playerList: Omit<PlayerStats, 'display_name'>[] = stats.map(
        (s: AggregatedStat) => ({
          playerid: s.playerid,
          games_played: s.games_played,
          avg_points: s.total_points / s.games_played,
          avg_rebounds: s.total_rebounds / s.games_played,
          avg_assists: s.total_assists / s.games_played,
        })
      );

      const playerIds = playerList.map((p) => p.playerid);
      const { data: names } = await supabase
        .from('players')
        .select('nba_api_id, display_name')
        .in('nba_api_id', playerIds);

      const nameMap = new Map(
        names?.map((n) => [n.nba_api_id, n.display_name || 'Unknown']) || []
      );
      const playersWithNames: PlayerStats[] = playerList.map((p) => ({
        ...p,
        display_name: nameMap.get(p.playerid) || 'Unknown',
      }));

      setAllPlayers(playersWithNames);
      setLoading(false);
    }

    fetchAllPlayers();
  }, []);

  const filteredPlayers = useMemo(() => {
    if (!searchTerm) return allPlayers;
    const term = searchTerm.toLowerCase();
    return allPlayers.filter((p) =>
      p.display_name.toLowerCase().includes(term)
    );
  }, [allPlayers, searchTerm]);

  const sortedPlayers = useMemo(() => {
    const sorted = [...filteredPlayers];
    sorted.sort((a, b) => {
      const aVal = a[sortBy];
      const bVal = b[sortBy];
      if (sortDesc) return bVal - aVal;
      return aVal - bVal;
    });
    return sorted;
  }, [filteredPlayers, sortBy, sortDesc]);

  const handleSort = (key: SortKey) => {
    if (sortBy === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortBy(key);
      setSortDesc(key === 'avg_points');
    }
  };

  const clearSearch = () => setSearchTerm('');

  const statColumns: { key: SortKey; label: string; title: string }[] = [
    { key: 'avg_points',   label: 'PPG', title: 'Points per game'   },
    { key: 'avg_rebounds', label: 'RPG', title: 'Rebounds per game' },
    { key: 'avg_assists',  label: 'APG', title: 'Assists per game'  },
    { key: 'games_played', label: 'GP',  title: 'Games played'      },
  ];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;500;600;700&family=Barlow:wght@400;500&display=swap');
        .players-root { min-height:100vh; background:#080a0e; color:#e8eaf0; font-family:'Barlow',sans-serif; padding:2rem 1.5rem 4rem; }
        .players-inner { max-width:1100px; margin:0 auto; }
        .header-row { display:flex; align-items:flex-start; gap:1.25rem; margin-bottom:2.5rem; }
        .back-btn { display:flex; align-items:center; justify-content:center; width:2.25rem; height:2.25rem; border-radius:6px; background:#111318; border:1px solid #1e2230; color:#7a7f96; text-decoration:none; flex-shrink:0; margin-top:4px; transition:background .15s,border-color .15s,color .15s; }
        .back-btn:hover { background:#1a1d26; border-color:#f05a1a; color:#f05a1a; }
        .header-eyebrow { font-family:'Barlow Condensed',sans-serif; font-size:0.7rem; font-weight:600; letter-spacing:.15em; text-transform:uppercase; color:#f05a1a; margin-bottom:.25rem; }
        .header-title { font-family:'Barlow Condensed',sans-serif; font-size:2.4rem; font-weight:700; letter-spacing:-.01em; line-height:1; color:#f0f2f7; margin:0; }
        .header-sub { font-size:.85rem; color:#4e5470; margin-top:.35rem; }
        .search-wrap { position:relative; margin-bottom:1rem; }
        .search-icon { position:absolute; left:.875rem; top:50%; transform:translateY(-50%); color:#4e5470; pointer-events:none; width:1rem; height:1rem; }
        .search-input { width:100%; padding:.7rem 2.5rem .7rem 2.5rem; background:#0d0f15; border:1px solid #1e2230; border-radius:8px; color:#e8eaf0; font-family:'Barlow',sans-serif; font-size:.9rem; outline:none; transition:border-color .15s,box-shadow .15s; box-sizing:border-box; }
        .search-input::placeholder { color:#3a3f58; }
        .search-input:focus { border-color:#f05a1a; box-shadow:0 0 0 3px rgba(240,90,26,.1); }
        .search-clear { position:absolute; right:.875rem; top:50%; transform:translateY(-50%); background:none; border:none; color:#4e5470; cursor:pointer; font-size:.85rem; padding:0; transition:color .15s; }
        .search-clear:hover { color:#e8eaf0; }
        .player-count { font-size:.78rem; color:#3a3f58; margin-bottom:1rem; font-family:'Barlow Condensed',sans-serif; letter-spacing:.06em; text-transform:uppercase; }
        .player-count span { color:#6b7090; }
        .loading-wrap { display:flex; flex-direction:column; align-items:center; justify-content:center; height:14rem; gap:1rem; }
        .spinner { width:2rem; height:2rem; border:2px solid #1e2230; border-top-color:#f05a1a; border-radius:50%; animation:spin .7s linear infinite; }
        @keyframes spin { to { transform:rotate(360deg); } }
        .loading-label { font-family:'Barlow Condensed',sans-serif; font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; color:#3a3f58; }
        .error-box { background:#130a08; border:1px solid #3d1a10; border-left:3px solid #e03d1a; border-radius:8px; padding:1rem 1.25rem; }
        .error-title { font-family:'Barlow Condensed',sans-serif; font-weight:600; font-size:.95rem; color:#e8604a; margin-bottom:.25rem; }
        .error-msg { font-size:.82rem; color:#9a5040; }
        .empty-state { text-align:center; padding:5rem 0; color:#3a3f58; font-family:'Barlow Condensed',sans-serif; font-size:1rem; letter-spacing:.04em; }
        .empty-state strong { color:#5a6080; }
        .table-container { border:1px solid #1e2230; border-radius:10px; overflow:hidden; }
        .table-scroll { overflow-x:auto; }
        table { width:100%; border-collapse:collapse; min-width:520px; }
        thead tr { background:#0d0f15; border-bottom:1px solid #1e2230; }
        th { padding:.75rem 1.25rem; font-family:'Barlow Condensed',sans-serif; font-size:.72rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:#3a3f58; white-space:nowrap; user-select:none; }
        th.sortable { cursor:pointer; transition:color .15s; }
        th.sortable:hover { color:#8a90a8; }
        th.sortable.active { color:#f05a1a; }
        th.col-name { text-align:left; }
        th.col-stat { text-align:right; }
        .th-inner { display:flex; align-items:center; justify-content:flex-end; gap:.3rem; }
        .th-inner.left { justify-content:flex-start; }
        .sort-arrow { font-size:.65rem; opacity:.9; color:#f05a1a; }
        tbody tr { border-bottom:1px solid #111318; cursor:pointer; transition:background .12s; }
        tbody tr:last-child { border-bottom:none; }
        tbody tr:hover { background:#111520; }
        tbody tr:hover .cell-name { color:#f05a1a; }
        tbody tr:hover .cell-arrow { color:#f05a1a; }
        td { padding:.8rem 1.25rem; white-space:nowrap; }
        .cell-rank { font-family:'Barlow Condensed',sans-serif; font-size:.72rem; color:#2a2f48; width:2rem; text-align:right; padding-right:0; }
        .cell-name { font-family:'Barlow Condensed',sans-serif; font-size:1rem; font-weight:600; color:#d8dae8; letter-spacing:.01em; transition:color .12s; }
        .cell-arrow { font-size:.75rem; color:#2a2f48; margin-left:.35rem; transition:color .12s; }
        .cell-stat { text-align:right; font-family:'Barlow Condensed',sans-serif; font-size:.95rem; font-weight:500; color:#8a90a8; letter-spacing:.02em; }
        .cell-stat.active-col { color:#e8eaf0; }
        .cell-gp { text-align:right; font-family:'Barlow Condensed',sans-serif; font-size:.88rem; color:#3a3f58; }
        .cell-gp.active-col { color:#8a90a8; }
        tbody tr:first-child .cell-stat.active-col { color:#f05a1a; }
      `}</style>

      <main className="players-root">
        <div className="players-inner">

          <div className="header-row">
            <Link href="/" className="back-btn" aria-label="Back to home">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </Link>
            <div>
              <div className="header-eyebrow">Statistics</div>
              <h1 className="header-title">NBA Players</h1>
              <p className="header-sub">2025–26 Regular Season</p>
            </div>
          </div>

          <div className="search-wrap">
            <svg className="search-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Search players..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
            {searchTerm && (
              <button onClick={clearSearch} className="search-clear">✕</button>
            )}
          </div>

          {!loading && !errorMsg && (
            <div className="player-count">
              <span>{sortedPlayers.length}</span> of <span>{allPlayers.length}</span> players
            </div>
          )}

          {loading && (
            <div className="loading-wrap">
              <div className="spinner" />
              <span className="loading-label">Loading players</span>
            </div>
          )}

          {errorMsg && (
            <div className="error-box">
              <div className="error-title">Failed to load players</div>
              <div className="error-msg">{errorMsg}</div>
            </div>
          )}

          {!loading && !errorMsg && sortedPlayers.length === 0 && (
            <div className="empty-state">
              No players matching <strong>&quot;{searchTerm}&quot;</strong>
            </div>
          )}

          {!loading && !errorMsg && sortedPlayers.length > 0 && (
            <div className="table-container">
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th className="col-name" style={{ width: '2rem', paddingRight: 0 }} />
                      <th className="col-name">
                        <div className="th-inner left">Player</div>
                      </th>
                      {statColumns.map(({ key, label, title }) => (
                        <th
                          key={key}
                          className={`col-stat sortable${sortBy === key ? ' active' : ''}`}
                          title={title}
                          onClick={(e) => { e.stopPropagation(); handleSort(key); }}
                        >
                          <div className="th-inner">
                            {label}
                            {sortBy === key && (
                              <span className="sort-arrow">{sortDesc ? '↓' : '↑'}</span>
                            )}
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedPlayers.map((player, idx) => (
                      <tr
                        key={player.playerid}
                        onClick={() => router.push(`/players/${player.playerid}`)}
                      >
                        <td className="cell-rank">{idx + 1}</td>
                        <td className="cell-name">
                          {player.display_name}
                          <span className="cell-arrow">›</span>
                        </td>
                        <td className={`cell-stat${sortBy === 'avg_points' ? ' active-col' : ''}`}>
                          {player.avg_points.toFixed(1)}
                        </td>
                        <td className={`cell-stat${sortBy === 'avg_rebounds' ? ' active-col' : ''}`}>
                          {player.avg_rebounds.toFixed(1)}
                        </td>
                        <td className={`cell-stat${sortBy === 'avg_assists' ? ' active-col' : ''}`}>
                          {player.avg_assists.toFixed(1)}
                        </td>
                        <td className={`cell-gp${sortBy === 'games_played' ? ' active-col' : ''}`}>
                          {player.games_played}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        </div>
      </main>
    </>
  );
}