const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

async function fetcher<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  getGames:       () => fetcher<any>('/api/games'),
  getTeams:       () => fetcher<any>('/api/teams'),
  getPredictions: () => fetcher<any>('/api/predictions'),
};