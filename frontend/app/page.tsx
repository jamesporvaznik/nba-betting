import Link from 'next/link';

export default function Home() {
  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">NBA Betting ML</h1>
      <p className="text-gray-500 mb-8">
        Machine learning predictions for NBA games
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Link href="/games"
          className="p-6 border rounded-lg hover:bg-gray-50 transition">
          <h2 className="text-lg font-semibold mb-1">Games</h2>
          <p className="text-gray-500 text-sm">Browse game history and upcoming matchups</p>
        </Link>
        <Link href="/predictions"
          className="p-6 border rounded-lg hover:bg-gray-50 transition">
          <h2 className="text-lg font-semibold mb-1">Predictions</h2>
          <p className="text-gray-500 text-sm">ML model predictions and confidence scores</p>
        </Link>
        <Link href="/teams"
          className="p-6 border rounded-lg hover:bg-gray-50 transition">
          <h2 className="text-lg font-semibold mb-1">Teams</h2>
          <p className="text-gray-500 text-sm">Team stats and performance trends</p>
        </Link>
      </div>
    </main>
  );
}