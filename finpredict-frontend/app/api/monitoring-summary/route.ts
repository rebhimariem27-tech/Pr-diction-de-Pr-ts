import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET() {
  const apiBase = (process.env.FINPREDICT_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const apiKey = process.env.FINPREDICT_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ detail: 'FINPREDICT_API_KEY n’est pas configurée côté serveur Next.js.' }, { status: 500 });
  }
  try {
    const response = await fetch(`${apiBase}/monitoring-summary`, {
      headers: { Accept: 'application/json', 'X-API-Key': apiKey },
      cache: 'no-store',
      signal: AbortSignal.timeout(30_000),
    });
    const body = await response.json().catch(() => ({ detail: 'Réponse monitoring invalide.' }));
    return NextResponse.json(body, { status: response.status });
  } catch {
    return NextResponse.json({ detail: `Impossible de joindre l’API à ${apiBase}.` }, { status: 502 });
  }
}
